#!/usr/bin/env python3
"""
exp_7178_1d_donchian20_1w_hma_v2
Hypothesis: 1d Donchian(20) breakout with 1w HMA(21) trend filter + volume confirmation.
In trending markets (price > 1w HMA): take continuation breakouts only.
In ranging markets (price near 1w HMA): take mean reversion at Donchian extremes.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-30 trades/year.
Works in bull/bear by adapting to HMA-defined regime. Includes ATR stoploss and time-based exit.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7178_1d_donchian20_1w_hma_v2"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
HMA_PERIOD = 21
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0  # Increased to reduce trades
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 15  # Increased to reduce trade frequency

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for HMA trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA (Hull Moving Average)
    close_1w = df_1w['close'].values
    half_period = HMA_PERIOD // 2
    sqrt_period = int(np.sqrt(HMA_PERIOD))
    
    # WMA function with proper min_periods handling
    def wma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    # HMA calculation
    wma_half = wma(close_1w, half_period)
    wma_full = wma(close_1w, HMA_PERIOD)
    
    # Handle array lengths
    if len(wma_half) > 0 and len(wma_full) > 0:
        raw_hma = 2 * wma_half - wma_full
        hma_values = wma(raw_hma, sqrt_period)
        # Pad to original length
        hma_1w = np.full_like(close_1w, np.nan)
        start_idx = half_period - 1
        end_idx = start_idx + len(hma_values)
        hma_1w[start_idx:end_idx] = hma_values
    else:
        hma_1w = np.full_like(close_1w, np.nan)
    
    # Align to LTF (1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels with min_periods
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, HMA_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation with higher threshold
        vol_confirmed = (not np.isnan(vol_ma[i]) and 
                        volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD)
        
        # Determine market regime based on HMA
        above_hma = close[i] > hma_1w_aligned[i]
        below_hma = close[i] < hma_1w_aligned[i]
        near_hma = np.abs(close[i] - hma_1w_aligned[i]) < (1.0 * atr[i])  # Within 1.0 ATR of HMA
        
        # Fade at extremes in ranging market (near HMA)
        fade_long = near_hma and (close[i] <= lowest_low[i]) and vol_confirmed
        fade_short = near_hma and (close[i] >= highest_high[i]) and vol_confirmed
        
        # Continuation breakouts in trending market
        continuation_long = above_hma and (close[i] > highest_high[i]) and vol_confirmed
        continuation_short = below_hma and (close[i] < lowest_low[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if fade_long or continuation_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif fade_short or continuation_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
exp_7178_1d_donchian20_1w_hma_v2
Hypothesis: 1d Donchian(20) breakout with 1w HMA(21) trend filter + volume confirmation.
In trending markets (price > 1w HMA): take continuation breakouts only.
In ranging markets (price near 1w HMA): take mean reversion at Donchian extremes.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-30 trades/year.
Works in bull/bear by adapting to HMA-defined regime. Includes ATR stoploss and time-based exit.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7178_1d_donchian20_1w_hma_v2"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
HMA_PERIOD = 21
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0  # Increased to reduce trades
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 15  # Increased to reduce trade frequency

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for HMA trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA (Hull Moving Average)
    close_1w = df_1w['close'].values
    half_period = HMA_PERIOD // 2
    sqrt_period = int(np.sqrt(HMA_PERIOD))
    
    # WMA function with proper min_periods handling
    def wma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    # HMA calculation
    wma_half = wma(close_1w, half_period)
    wma_full = wma(close_1w, HMA_PERIOD)
    
    # Handle array lengths
    if len(wma_half) > 0 and len(wma_full) > 0:
        raw_hma = 2 * wma_half - wma_full
        hma_values = wma(raw_hma, sqrt_period)
        # Pad to original length
        hma_1w = np.full_like(close_1w, np.nan)
        start_idx = half_period - 1
        end_idx = start_idx + len(hma_values)
        hma_1w[start_idx:end_idx] = hma_values
    else:
        hma_1w = np.full_like(close_1w, np.nan)
    
    # Align to LTF (1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels with min_periods
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, HMA_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation with higher threshold
        vol_confirmed = (not np.isnan(vol_ma[i]) and 
                        volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD)
        
        # Determine market regime based on HMA
        above_hma = close[i] > hma_1w_aligned[i]
        below_hma = close[i] < hma_1w_aligned[i]
        near_hma = np.abs(close[i] - hma_1w_aligned[i]) < (1.0 * atr[i])  # Within 1.0 ATR of HMA
        
        # Fade at extremes in ranging market (near HMA)
        fade_long = near_hma and (close[i] <= lowest_low[i]) and vol_confirmed
        fade_short = near_hma and (close[i] >= highest_high[i]) and vol_confirmed
        
        # Continuation breakouts in trending market
        continuation_long = above_hma and (close[i] > highest_high[i]) and vol_confirmed
        continuation_short = below_hma and (close[i] < lowest_low[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if fade_long or continuation_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif fade_short or continuation_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals