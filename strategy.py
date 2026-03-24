#!/usr/bin/env python3
"""
Experiment #717: 15m Primary + 4h/12h HTF — Fisher Transform + Choppiness Regime + Session Filter

Hypothesis: 15m timeframe with strict 4h/12h HTF filters can achieve optimal trade frequency (50-100/year)
while capturing intraday moves. Key innovations:
1. Ehlers Fisher Transform(9) - superior reversal detection vs RSI in bear/range markets
2. Choppiness Index(14) regime filter - only trade when CHOP indicates clear regime
3. 4h HMA(21) for primary trend bias
4. 12h HMA(48) for secondary confirmation (triple MTF)
5. Session filter - only trade 00-12 UTC (London+NY overlap, highest crypto volume)
6. Volume confirmation - taker_buy_volume ratio > 0.55 for longs, < 0.45 for shorts
7. ATR(14) 2.5x trailing stoploss
8. Discrete sizing: 0.0, ±0.15, ±0.20 (conservative for 15m frequency)

Entry confluence (ALL required):
- LONG: 4h HMA bull + 12h HMA bull + Fisher < -1.5 + CHOP < 50 (trending) OR CHOP > 55 (range mean-revert)
- SHORT: 4h HMA bear + 12h HMA bear + Fisher > +1.5 + CHOP < 50 OR CHOP > 55
- Session: hour 00-12 UTC only
- Volume: taker_buy_ratio confirms direction

Target: Sharpe>0.40, trades>=40 train, trades>=5 test, DD>-35%, trades/year < 100
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for 15m frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_fisher_chop_session_4h12h_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Catches reversals better than RSI in bear/range markets
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    for i in range(period, n):
        # Use (high + low) / 2 as price input
        hl2 = (high[i-period+1:i+1].max() + low[i-period+1:i+1].min()) / 2.0
        hl2_prev = (high[i-period:i].max() + low[i-period:i].min()) / 2.0 if i > period else hl2
        
        # Normalize to -1 to +1 range
        if hl2 == hl2_prev:
            continue
        
        xform = 0.66 * ((hl2 - low[i-period+1:i+1].min()) / (hl2 - hl2_prev + 1e-10) - 0.5) + 0.67 * (0.66 * ((hl2_prev - low[i-period:i].min()) / (hl2_prev - hl2 + 1e-10) - 0.5) if i > period else 0)
        xform = np.clip(xform, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + xform) / (1.0 - xform + 1e-10))
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher_prev[i] = fisher[i-1]
        else:
            fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - identifies trending vs ranging markets"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_taker_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio"""
    n = len(volume)
    ratio = np.zeros(n)
    ratio[:] = np.nan
    
    for i in range(n):
        if volume[i] > 1e-10:
            ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            ratio[i] = 0.5
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA (4h)
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align HTF HMA (12h)
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=48)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 15m indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    taker_ratio = calculate_taker_ratio(taker_buy_vol, volume)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = 0 <= hour_utc <= 12
        
        if not in_session:
            # Close existing position outside session
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h + 12h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        # Strong bias: both 4h and 12h agree
        htf_strong_bull = htf_4h_bull and htf_12h_bull
        htf_strong_bear = htf_4h_bear and htf_12h_bear
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP < 38.2 = strong trend, 38.2-61.8 = neutral, > 61.8 = range
        trend_regime = chop[i] < 50.0  # Loose for more trades
        range_regime = chop[i] > 55.0  # Loose for more trades
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        fisher_cross_down = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # === VOLUME CONFIRMATION ===
        vol_bull = taker_ratio[i] > 0.55
        vol_bear = taker_ratio[i] < 0.45
        
        # === ENTRY LOGIC (3+ CONFLUENCE REQUIRED) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + Fisher oversold/cross + (trend OR range) + volume
        if htf_strong_bull:
            # Trend regime: Fisher cross up + volume
            if trend_regime and fisher_cross_up and vol_bull:
                desired_signal = SIZE_STRONG
            # Range regime: Fisher oversold + volume (mean reversion)
            elif range_regime and fisher_oversold and vol_bull:
                desired_signal = SIZE_BASE
            # Very oversold Fisher (any regime)
            elif fisher[i] < -2.0 and vol_bull:
                desired_signal = SIZE_BASE
        
        # SHORT: HTF bear + Fisher overbought/cross + (trend OR range) + volume
        elif htf_strong_bear:
            # Trend regime: Fisher cross down + volume
            if trend_regime and fisher_cross_down and vol_bear:
                desired_signal = -SIZE_STRONG
            # Range regime: Fisher overbought + volume (mean reversion)
            elif range_regime and fisher_overbought and vol_bear:
                desired_signal = -SIZE_BASE
            # Very overbought Fisher (any regime)
            elif fisher[i] > 2.0 and vol_bear:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals