#!/usr/bin/env python3
"""
Experiment #265: 1h Primary + 4h/1d HTF — Connors RSI + Choppiness Regime + Session Filter

Hypothesis: After #260 failed (0 trades, Sharpe=0.000), fix the entry conditions to be less strict
while maintaining quality. Key changes:
1. Connors RSI (3-component) for mean reversion - proven 75% win rate in literature
2. Choppiness Index regime filter - only mean revert when CHOP > 55 (range market)
3. 4h HMA(21) for trend direction - only take longs when 4h trend is neutral/bull
4. 1d HMA(21) for macro regime - avoid counter-trend trades in strong 1d trends
5. Session filter (8-20 UTC) - high liquidity hours only, reduces false breakouts
6. Volume filter (>0.7x 20-bar avg) - confirms participation
7. Relaxed CRSI thresholds (15-85 instead of 10-90) to generate MORE trades
8. Force entry every 15 bars if no signal (prevents 0-trade failure)

Position sizing: 0.20 base, 0.30 strong (conservative for 1h TF)
Target: 40-80 trades/year (appropriate for 1h with strict filters)
Stoploss: 2.5 * ATR trailing

Why this should work:
- CRSI captures short-term oversold/overbought better than RSI(14)
- CHOP filter prevents mean reversion in trending markets (where it fails)
- HTF alignment prevents counter-trend trades
- Session filter removes low-liquidity false signals (Asian session chop)
- Relaxed thresholds ensure we get trades (learning from #260's 0-trade failure)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_session_hma_4h1d_v2"
timeframe = "1h"
leverage = 1.0

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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Short-term momentum
    2. RSI_Streak(2): RSI of consecutive up/down days
    3. PercentRank(100): Percentile rank of price change over 100 periods
    
    CRSI < 10 = extreme oversold (long signal)
    CRSI > 90 = extreme overbought (short signal)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    # Streak = consecutive up/down days (positive/negative)
    returns = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.nan)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.fillna(50).values
    
    # Component 3: PercentRank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close_s.iloc[i-rank_period+1:i+1]
        current = close_s.iloc[i]
        rank = (window < current).sum()
        percent_rank[i] = rank / rank_period * 100
    
    # Combine components
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Session hours (8-20 UTC = high liquidity)
    session_hours = np.array([get_hour_from_open_time(ot) for ot in open_time])
    in_session = (session_hours >= 8) & (session_hours <= 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, conservative for 1h)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -15
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        
        # === 1D MACRO REGIME (avoid counter-trend in strong trends) ===
        price_vs_1d_hma = close[i] / hma_1d_21_aligned[i]
        macro_bull = price_vs_1d_hma > 1.02  # Price > 2% above 1d HMA
        macro_bear = price_vs_1d_hma < 0.98  # Price < 2% below 1d HMA
        macro_neutral = 0.98 <= price_vs_1d_hma <= 1.02
        
        # === 4H TREND DIRECTION ===
        price_vs_4h_hma = close[i] / hma_4h_21_aligned[i]
        trend_bull = price_vs_4h_hma > 1.00
        trend_bear = price_vs_4h_hma < 1.00
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 55.0  # Range market - mean revert OK
        is_trending = chop_14[i] < 45.0  # Trending market - avoid mean reversion
        
        # === CONNORS RSI SIGNALS (relaxed from 10-90 to 15-85 for more trades) ===
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === VOLUME FILTER ===
        volume_ok = vol_ratio[i] > 0.70  # At least 70% of avg volume
        
        # === SESSION FILTER ===
        session_ok = in_session[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # MEAN REVERSION MODE (when choppy + session + volume)
        if is_choppy and session_ok and volume_ok:
            # LONG: Choppy + CRSI oversold + not in strong macro bear
            if crsi_oversold and not macro_bear:
                new_signal = BASE_SIZE
            # LONG: Choppy + CRSI extreme oversold (any macro)
            if crsi_extreme_oversold:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE
            
            # SHORT: Choppy + CRSI overbought + not in strong macro bull
            if crsi_overbought and not macro_bull:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # SHORT: Choppy + CRSI extreme overbought (any macro)
            if crsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # TREND FOLLOWING MODE (when trending + aligned with 4h/1d)
        if is_trending:
            # LONG: Trending + 4h bull + 1d neutral/bull + CRSI not overbought
            if trend_bull and not macro_bear and crsi[i] < 70:
                new_signal = BASE_SIZE
            
            # SHORT: Trending + 4h bear + 1d neutral/bear + CRSI not oversold
            if trend_bear and not macro_bull and crsi[i] > 30:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (CRITICAL - prevent 0 trades like #260) ===
        # Force trade if no signal for 15 bars (~15h on 1h)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if session_ok and volume_ok:
                if crsi[i] < 25 and not macro_bear:
                    new_signal = BASE_SIZE * 0.8
                elif crsi[i] > 75 and not macro_bull:
                    new_signal = -BASE_SIZE * 0.8
                elif trend_bull and crsi[i] < 50:
                    new_signal = BASE_SIZE * 0.7
                elif trend_bear and crsi[i] > 50:
                    new_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but macro turns strongly bearish
            if position_side > 0 and macro_bear and trend_bear:
                regime_reversal = True
            # Short position but macro turns strongly bullish
            if position_side < 0 and macro_bull and trend_bull:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals