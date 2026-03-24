#!/usr/bin/env python3
"""
Experiment #546: 1d Primary + 1w HTF — Fisher Transform Reversals + HMA Trend + Volume

Hypothesis: Daily timeframe with Ehlers Fisher Transform provides superior reversal
detection in bear/range markets (2022 crash, 2025 bear). Fisher normalizes price to
Gaussian distribution, making extremes at -2/+2 highly reliable reversal signals.
Combined with 1w HMA for macro bias and volume confirmation, this should generate
20-40 trades/year with high win rate.

Key differences from failed experiments:
1. Fisher Transform instead of RSI/CRSI - better for bear market reversals
2. Simpler entry logic - only 3 conditions (HTF bias + Fisher extreme + volume)
3. Fewer regime filters - less conflict = more trades generated
4. 1d timeframe - proven to work better than 4h/6h for trend/reversal strategies
5. Volume confirmation - ensures breakouts/reversals have participation

Strategy logic:
1. 1w HMA(21) = macro trend bias (price above = bull bias, below = bear bias)
2. 1d Fisher Transform(9) = reversal signals (cross above -1.5 = long, cross below +1.5 = short)
3. 1d Volume ratio(20) = confirm participation (vol > 1.2x avg = valid signal)
4. 1d ATR(14)*2.5 = stoploss on all positions
5. 1d ADX(14) = optional trend strength filter (ADX>20 = trend valid)

Entry conditions (LOOSE to ensure trades):
- Long: Price > 1w HMA + Fisher crosses above -1.5 + Volume > 1.0x avg
- Short: Price < 1w HMA + Fisher crosses below +1.5 + Volume > 1.0x avg
- Exit: Fisher crosses opposite threshold OR stoploss hit

Target: Sharpe>0.40, trades>=30 train (7.5/year), trades>=5 test
Timeframe: 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_hma_vol_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution for clearer reversal signals
    
    Steps:
    1. Calculate typical price = (high + low) / 2
    2. Normalize: (price - lowest_low) / (highest_high - lowest_low)
    3. Scale to -1 to +1 range
    4. Apply Fisher: 0.5 * ln((1+x)/(1-x))
    
    Signals:
    - Fisher crosses above -1.5 from below = long reversal
    - Fisher crosses below +1.5 from above = short reversal
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    # Calculate typical price
    typical = (high + low) / 2.0
    
    for i in range(period, n):
        # Find highest high and lowest low over lookback
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        
        if price_range < 1e-10:
            fisher[i] = 0.0
            continue
        
        # Normalize price to 0-1 range
        normalized = (typical[i] - lowest_low) / price_range
        
        # Scale to -0.99 to +0.99 (avoid division by zero in ln)
        scaled = 2.0 * normalized - 1.0
        scaled = np.clip(scaled, -0.99, 0.99)
        
        # Apply Fisher transform
        if abs(scaled) < 0.999:
            fisher[i] = 0.5 * np.log((1.0 + scaled) / (1.0 - scaled))
        else:
            fisher[i] = np.sign(scaled) * 2.0
    
    return fisher

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

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    fisher = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = -0.30
    SIZE_HALF = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Fisher cross tracking
    prev_fisher = np.nan
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        if np.isnan(fisher[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        # === HTF BIAS (1w macro) ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 1e-10 else 1.0
        vol_confirmed = vol_ratio > 1.0  # Loose filter to ensure trades
        
        # === ADX TREND STRENGTH ===
        adx_valid = adx[i] > 15.0 if not np.isnan(adx[i]) else True  # Very loose
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = False
        fisher_cross_short = False
        
        if not np.isnan(prev_fisher) and not np.isnan(fisher[i]):
            # Long: Fisher crosses above -1.5 from below
            if prev_fisher < -1.5 and fisher[i] >= -1.5:
                fisher_cross_long = True
            # Short: Fisher crosses below +1.5 from above
            if prev_fisher > 1.5 and fisher[i] <= 1.5:
                fisher_cross_short = True
        
        # === ENTRY LOGIC (LOOSE to ensure trades) ===
        desired_signal = 0.0
        
        # Long entry: HTF bull + Fisher cross long + volume confirmed
        if htf_bull and fisher_cross_long and vol_confirmed:
            desired_signal = SIZE_LONG
        
        # Short entry: HTF bear + Fisher cross short + volume confirmed
        elif htf_bear and fisher_cross_short and vol_confirmed:
            desired_signal = SIZE_SHORT
        
        # Additional: Fisher extreme reversal (even without cross)
        if desired_signal == 0.0:
            # Very oversold in bull market
            if htf_bull and fisher[i] < -2.0 and vol_confirmed:
                desired_signal = SIZE_HALF
            # Very overbought in bear market
            elif htf_bear and fisher[i] > 2.0 and vol_confirmed:
                desired_signal = -SIZE_HALF
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT ON OPPOSITE FISHER SIGNAL ===
        if in_position and position_side > 0 and fisher[i] > 1.5:
            desired_signal = 0.0  # Take profit on long
        if in_position and position_side < 0 and fisher[i] < -1.5:
            desired_signal = 0.0  # Take profit on short
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_LONG * 0.9:
            final_signal = SIZE_LONG
        elif desired_signal <= SIZE_SHORT * 0.9:
            final_signal = SIZE_SHORT
        elif desired_signal >= SIZE_HALF * 0.9:
            final_signal = SIZE_HALF
        elif desired_signal <= -SIZE_HALF * 0.9:
            final_signal = -SIZE_HALF
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
        
        # Update prev_fisher for next iteration
        prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
    
    return signals