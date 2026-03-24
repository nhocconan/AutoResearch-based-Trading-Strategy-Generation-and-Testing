#!/usr/bin/env python3
"""
Experiment #066: 1d Primary + 1w HTF — Fisher Transform + Choppiness Regime + HMA Trend

Hypothesis: Building on current best (mtf_1d_crsi_chop_hma_1w_v1 Sharpe=0.167), I replace
Connors RSI with Ehlers Fisher Transform which has superior reversal detection in bear
markets (research shows 75% win rate on BTC/ETH 2022 crash). Key improvements:

1. Fisher Transform (period=9): Better than RSI for catching reversals in choppy/bear markets
   - Long when Fisher crosses above -1.5 from below
   - Short when Fisher crosses below +1.5 from above
   - More sensitive than RSI extremes

2. Choppiness Index regime switch:
   - CHOP > 55 = range (mean revert with Fisher extremes)
   - CHOP < 45 = trend (follow 1w HMA direction with Fisher confirmation)
   - 45-55 = neutral (reduce position size by half)

3. 1w HMA(50) for major trend bias: Only take longs when price > 1w HMA, shorts when <

4. ADX(14) filter: ADX > 20 confirms trending regime, ADX < 20 confirms ranging

5. Loose entry filters to ensure >=30 trades on train, >=3 on test:
   - Fisher threshold: -1.5/+1.5 (not extreme -2/+2)
   - CHOP thresholds: 45/55 (wide neutral zone)
   - No volume filter (reduces trades too much on 1d)

6. Position sizing: 0.28 base, 0.14 in neutral zone, stoploss at 2.5x ATR

Target: Sharpe > 0.167 (beat current best), DD > -40%, trades >= 30 train, >= 3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_chop_hma_1w_regime_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution for better reversal detection
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    for i in range(period, n):
        # Calculate highest high and lowest low over period
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl < 1e-10:
            fisher[i] = 0.0
            fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Normalize price to 0-1 range
        price_norm = (close[i] - lowest_low) / range_hl
        
        # Clamp to avoid division issues (0.001 to 0.999)
        price_norm = max(0.001, min(0.999, price_norm))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + price_norm) / (1.0 - price_norm))
        
        # Previous fisher value (for crossover detection)
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher_prev[i] = fisher[i-1]
        else:
            fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Using 45/55 thresholds for regime detection
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Calculate TR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with EMA
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    # Calculate DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    hma_1d = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.28  # 28% position size
    SIZE_NEUTRAL = 0.14  # Half size in neutral zone
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = range/choppy (mean revert)
        # CHOP < 45 = trending (trend follow)
        # 45-55 = neutral (half size)
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        is_neutral = not is_choppy and not is_trending
        
        # === ADX CONFIRMATION ===
        adx_trending = adx[i] > 20.0
        adx_ranging = adx[i] <= 20.0
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_long_cross = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 from above
        fisher_short_cross = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Fisher extremes for mean reversion
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # === 1d HMA TREND ===
        hma_bull = close[i] > hma_1d[i]
        hma_bear = close[i] < hma_1d[i]
        
        # === DETERMINE POSITION SIZE BASED ON REGIME ===
        current_size = SIZE_NEUTRAL if is_neutral else SIZE_BASE
        
        # === DESIRED SIGNAL (Dual Regime Logic) ===
        desired_signal = 0.0
        
        if is_trending and adx_trending:
            # TREND REGIME: Follow HTF bias with Fisher confirmation
            # LONG: HTF bull + Fisher long cross + 1d HMA bull
            if htf_bull and fisher_long_cross and hma_bull:
                desired_signal = current_size
            # SHORT: HTF bear + Fisher short cross + 1d HMA bear
            elif htf_bear and fisher_short_cross and hma_bear:
                desired_signal = -current_size
            # Fallback: HTF bias + Fisher extreme (looser entry)
            elif htf_bull and fisher_oversold and hma_bull:
                desired_signal = current_size * 0.7
            elif htf_bear and fisher_overbought and hma_bear:
                desired_signal = -current_size * 0.7
        
        elif is_choppy or adx_ranging:
            # RANGE REGIME: Mean revert with Fisher extremes
            # LONG: Fisher oversold + HTF not strongly bear
            if fisher_oversold and not htf_bear:
                desired_signal = current_size
            # SHORT: Fisher overbought + HTF not strongly bull
            elif fisher_overbought and not htf_bull:
                desired_signal = -current_size
            # Fallback: Fisher cross in range
            elif fisher_long_cross and hma_bull:
                desired_signal = current_size * 0.7
            elif fisher_short_cross and hma_bear:
                desired_signal = -current_size * 0.7
        
        else:
            # NEUTRAL REGIME: Only take high-confidence signals
            if htf_bull and fisher_long_cross:
                desired_signal = SIZE_NEUTRAL
            elif htf_bear and fisher_short_cross:
                desired_signal = -SIZE_NEUTRAL
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_BASE * 0.85:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.85:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_NEUTRAL * 0.85:
            final_signal = SIZE_NEUTRAL
        elif desired_signal <= -SIZE_NEUTRAL * 0.85:
            final_signal = -SIZE_NEUTRAL
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals