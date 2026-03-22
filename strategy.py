#!/usr/bin/env python3
"""
Experiment #620: 1h Primary + 4h/12h HTF — HMA Trend + Connors RSI + Choppiness Regime + Session Filter

Hypothesis: After 548 failed strategies, key insight is lower TF needs STRICT entry filters
to avoid fee drag. This strategy uses:
1. 4h HMA(21) for primary trend direction (proven in best strategies)
2. 12h ADX(14) for trend strength (ADX>25 trend, ADX<20 range)
3. 1h Connors RSI for entry timing (3-component: RSI(3) + StreakRSI(2) + PercentRank(100))
4. Choppiness Index regime (CHOP>55 mean-revert, CHOP<45 trend-follow)
5. Session filter (8-20 UTC only - avoids Asian session chop, 50% fewer bars)
6. Volume filter (volume > 0.7x 20-bar avg)
7. Conservative size 0.25 (lower TF = smaller per Rule 4)
8. 2.5*ATR trailing stop

Why this might beat Sharpe=0.520:
- 1h entries within 4h/12h trend = HTF frequency with LTF precision
- Session filter cuts bars by 50% = fewer trades, less fee drag
- Connors RSI more reliable than simple RSI (75% win rate in literature)
- ADX hysteresis prevents regime whipsaw
- Volume confirmation avoids false breakouts

Position sizing: 0.25 discrete (conservative for 1h TF)
Target: 40-80 trades/year (per Rule 10 for 1h)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_crsi_chop_session_4h12h_v1"
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
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) of close - short-term momentum
    2. RSI(2) of streak - streak duration (consecutive up/down bars)
    3. PercentRank(100) - where current return ranks vs last 100 bars
    
    CRSI < 10 = extremely oversold (long)
    CRSI > 90 = extremely overbought (short)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) of close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI(2) of streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi_raw = calculate_rsi(np.abs(streak) + 1e-10, streak_period)
    streak_rsi = np.where(streak >= 0, streak_rsi_raw, 100 - streak_rsi_raw)
    
    # Component 3: PercentRank(100)
    returns = close_s.pct_change().values
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        current = returns[i]
        if np.isfinite(current):
            rank = np.sum(window < current) / len(window)
            percent_rank[i] = rank * 100
    
    # Combine components
    crsi = (rsi_close + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    More responsive than EMA with less lag.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period).mean().values
    
    return adx, plus_di, minus_di

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    atr_vals = calculate_atr(high, low, close, 14)
    
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop, 0.0, 100.0)
    chop = np.nan_to_num(chop, nan=50.0)
    
    return chop

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return pd.to_datetime(open_time, unit='ms').dt.hour.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HMA for primary trend
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 12h ADX for trend strength
    adx_12h, plus_di_12h, minus_di_12h = calculate_adx(
        df_12h['high'].values, 
        df_12h['low'].values, 
        df_12h['close'].values, 
        period=14
    )
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    plus_di_12h_aligned = align_htf_to_ltf(prices, df_12h, plus_di_12h)
    minus_di_12h_aligned = align_htf_to_ltf(prices, df_12h, minus_di_12h)
    
    # Calculate 1h indicators
    hma_1h = calculate_hma(close, period=21)
    crsi_1h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    utc_hour = get_utc_hour(open_time)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, conservative for 1h)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss (separate from signals)
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(hma_1h[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(adx_12h_aligned[i]) or np.isnan(chop_1h[i]) or np.isnan(atr_1h[i]):
            continue
        if np.isnan(crsi_1h[i]) or np.isnan(vol_avg_20[i]):
            continue
        if atr_1h[i] == 0 or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= utc_hour[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] >= 0.7 * vol_avg_20[i]
        
        # === 4H TREND BIAS (HMA slope over 3 bars) ===
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-3] if i >= 3 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-3] if i >= 3 else False
        
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 12H TREND STRENGTH (ADX with hysteresis) ===
        adx_strong = adx_12h_aligned[i] > 25.0
        adx_weak = adx_12h_aligned[i] < 20.0
        
        di_bull = plus_di_12h_aligned[i] > minus_di_12h_aligned[i]
        di_bear = plus_di_12h_aligned[i] < minus_di_12h_aligned[i]
        
        # === 1H REGIME (Choppiness Index) ===
        is_trend_regime = chop_1h[i] < 45.0
        is_chop_regime = chop_1h[i] > 55.0
        
        # === ENTRY LOGIC (relaxed for trade frequency - Rule 9) ===
        new_signal = 0.0
        allow_entry = in_session and volume_ok
        
        if allow_entry:
            # --- TREND REGIME: Follow 4h/12h trend with 1h CRSI pullback ---
            if is_trend_regime and adx_strong:
                # LONG: 4h bull + 12h bull + CRSI oversold (<25)
                if hma_4h_slope_bull and price_above_hma_4h and di_bull:
                    if crsi_1h[i] < 25.0:
                        new_signal = POSITION_SIZE
                
                # SHORT: 4h bear + 12h bear + CRSI overbought (>75)
                elif hma_4h_slope_bear and price_below_hma_4h and di_bear:
                    if crsi_1h[i] > 75.0:
                        new_signal = -POSITION_SIZE
            
            # --- CHOP REGIME: Mean reversion at CRSI extremes ---
            elif is_chop_regime or adx_weak:
                # LONG: CRSI < 20 (oversold)
                if crsi_1h[i] < 20.0:
                    new_signal = POSITION_SIZE
                
                # SHORT: CRSI > 80 (overbought)
                elif crsi_1h[i] > 80.0:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_1h[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_1h[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_4h_slope_bear and price_below_hma_4h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_4h_slope_bull and price_above_hma_4h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position and prev_signal != 0.0:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals