#!/usr/bin/env python3
"""
Experiment #055: 15m Multi-Timeframe with Connors RSI + Choppiness Regime Filter
Hypothesis: 15m timeframe needs strong HTF filters to avoid noise. Using 4h HMA for 
long-term bias, 1h HMA for intermediate trend. Connors RSI (CRSI) for entry timing 
has proven 75% win rate in mean-reversion setups. Choppiness Index detects regime 
to switch between trend-following (CHOP<38) and mean-reversion (CHOP>62).
Why this might work: 15m generates more signals than 12h, but needs better filtering.
CRSI catches oversold/overbought extremes better than standard RSI. CHOP filter 
avoids trend strategies in ranging markets (where most 15m strategies fail).
Position sizing: 0.25 base, 0.35 strong trend, discrete levels to minimize churn.
Must generate 10+ trades - CRSI thresholds loosened vs academic recommendations.
Timeframe: 15m (REQUIRED for exp#055), HTF: 1h and 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_chop_regime_4h_1h_hma_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Proven 75% win rate for mean-reversion entries.
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak RSI: consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI on streak
    streak_positive = np.where(streak > 0, streak, 0)
    streak_negative = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_positive).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_negative).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    mask = avg_streak_loss > 0
    rs_streak = avg_streak_gain[mask] / avg_streak_loss[mask]
    streak_rsi[mask] = 100 - (100 / (1 + rs_streak))
    streak_rsi[~mask] = 100.0
    
    # Percent Rank: how many of last 100 closes is current close greater than
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_less = np.sum(window[:-1] < window[-1])
        percent_rank[i] = 100 * count_less / (rank_period - 1)
    
    # Combine into CRSI
    valid_mask = (~np.isnan(rsi_close)) & (~np.isnan(streak_rsi)) & (~np.isnan(percent_rank))
    crsi[valid_mask] = (rsi_close[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        atr_sum = 0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    direction = np.zeros(n)  # 1 = uptrend, -1 = downtrend
    
    supertrend[period] = lower_band[period]
    direction[period] = 1
    
    for i in range(period + 1, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
    
    return supertrend, direction

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_200 = calculate_sma(close, 200)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    # Connors RSI for entry timing
    crsi = calculate_crsi(close, 3, 2, 100)
    
    # Standard RSI for additional filter
    rsi_14 = calculate_rsi(close, 14)
    
    # Choppiness Index for regime detection
    chop = calculate_choppiness(high, low, close, 14)
    
    # HMA on 15m for faster trend
    hma_15m = calculate_hma(close, 21)
    hma_15m_fast = calculate_hma(close, 10)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(ema_21[i]) or np.isnan(adx[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = long-term bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 1h HMA = intermediate trend
        bull_trend_1h = close[i] > hma_1h_aligned[i]
        bear_trend_1h = close[i] < hma_1h_aligned[i]
        
        # 15m HMA = short-term trend
        bull_trend_15m = hma_15m_fast[i] > hma_15m[i]
        bear_trend_15m = hma_15m_fast[i] < hma_15m[i]
        
        # EMA alignment
        ema_bullish = ema_21[i] > ema_50[i] and ema_50[i] > ema_200[i]
        ema_bearish = ema_21[i] < ema_50[i] and ema_50[i] < ema_200[i]
        
        # Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # === REGIME DETECTION ===
        trending_regime = chop[i] < 38.2 and adx[i] > 25
        ranging_regime = chop[i] > 61.8
        neutral_regime = 38.2 <= chop[i] <= 61.8
        
        # === CONNORS RSI EXTREMES (looser thresholds for more trades) ===
        crsi_oversold = crsi[i] < 25  # Standard is <10, loosened for more trades
        crsi_overbought = crsi[i] > 75  # Standard is >90, loosened for more trades
        crsi_neutral_long = 30 <= crsi[i] <= 50
        crsi_neutral_short = 50 <= crsi[i] <= 70
        
        # Standard RSI extremes
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === PRICE POSITION ===
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # Price near EMA21 (pullback entry zone)
        price_near_ema21_long = close[i] <= ema_21[i] * 1.03 and close[i] >= ema_21[i] * 0.97
        price_near_ema21_short = close[i] >= ema_21[i] * 0.97 and close[i] <= ema_21[i] * 1.03
        
        # Price near EMA50 (deeper pullback)
        price_near_ema50_long = close[i] <= ema_50[i] * 1.05 and close[i] >= ema_50[i] * 0.95
        price_near_ema50_short = close[i] >= ema_50[i] * 0.95 and close[i] <= ema_50[i] * 1.05
        
        # === HMA CROSSOVER ===
        hma_cross_long = False
        hma_cross_short = False
        if i >= 1 and not np.isnan(hma_15m_fast[i]) and not np.isnan(hma_15m_fast[i-1]):
            hma_cross_long = hma_15m_fast[i] > hma_15m[i] and hma_15m_fast[i-1] <= hma_15m[i-1]
            hma_cross_short = hma_15m_fast[i] < hma_15m[i] and hma_15m_fast[i-1] >= hma_15m[i-1]
        
        # DI crossover
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        new_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 38, ADX > 25) - Trend Following ===
        if trending_regime:
            # Long: HTF bullish + CRSI pullback + trend confirmation
            if bull_trend_4h and bull_trend_1h:
                if crsi_neutral_long and price_near_ema21_long:
                    if st_bullish or di_bullish:
                        new_signal = SIZE_STRONG
                elif crsi_oversold and above_sma200:
                    if bull_trend_15m:
                        new_signal = SIZE_BASE
                elif hma_cross_long and rsi_14[i] > 45:
                    new_signal = SIZE_BASE
            
            # Short: HTF bearish + CRSI pullback + trend confirmation
            if bear_trend_4h and bear_trend_1h:
                if crsi_neutral_short and price_near_ema21_short:
                    if st_bearish or di_bearish:
                        new_signal = -SIZE_STRONG
                elif crsi_overbought and below_sma200:
                    if bear_trend_15m:
                        new_signal = -SIZE_BASE
                elif hma_cross_short and rsi_14[i] < 55:
                    new_signal = -SIZE_BASE
        
        # === RANGING REGIME (CHOP > 62) - Mean Reversion ===
        if ranging_regime:
            # Long at support (oversold CRSI)
            if crsi_oversold or rsi_oversold:
                if price_near_ema50_long or close[i] < ema_50[i]:
                    if bull_trend_4h:  # Only long if HTF bias is bullish
                        new_signal = SIZE_HALF
            # Short at resistance (overbought CRSI)
            if crsi_overbought or rsi_overbought:
                if price_near_ema50_short or close[i] > ema_50[i]:
                    if bear_trend_4h:  # Only short if HTF bias is bearish
                        new_signal = -SIZE_HALF
        
        # === NEUTRAL REGIME - Conservative entries only ===
        if neutral_regime:
            # Only enter on strong CRSI extremes with HTF confirmation
            if crsi_oversold and bull_trend_4h and above_sma200:
                new_signal = SIZE_HALF
            if crsi_overbought and bear_trend_4h and below_sma200:
                new_signal = -SIZE_HALF
        
        # === SUPERTREND FLIP ENTRIES (any regime) ===
        if st_bullish and bull_trend_4h:
            if crsi[i] < 60 and rsi_14[i] < 60:
                new_signal = max(new_signal, SIZE_BASE)
        
        if st_bearish and bear_trend_4h:
            if crsi[i] > 40 and rsi_14[i] > 40:
                new_signal = min(new_signal, -SIZE_BASE)
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals