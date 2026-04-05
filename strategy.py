#!/usr/bin/env python3
"""
Experiment #7948: 12-hour Camarilla pivot with 1-week trend filter and volume confirmation.
Hypothesis: Price reacting from weekly Camarilla pivot levels (L3/H3) with volume >1.5x 24-period MA 
and aligned 1week trend (price above/below 1week EMA50) captures mean-reversion in ranging markets 
and continuation in trending markets. Weekly timeframe provides strong trend context to avoid 
counter-trend trades. Target: 50-150 total trades over 4 years.
"""

from mtf_data import get_ftf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7948_12h_camarilla1w_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # daily OHLC for pivot calculation
VOLUME_MA_PERIOD = 24  # 24 * 12h = 12 days
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
EMA_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1week EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Price relative to EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_1w > ema_1w, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1w, price_vs_ema)
    
    # Calculate daily OHLC for Camarilla pivot (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
    # H2 = C + 1.1*(H-L)/6, L2 = C - 1.1*(H-L)/6
    # H1 = C + 1.1*(H-L)/12, L1 = C - 1.1*(H-L)/12
    rang = high_1d - low_1d
    h4 = close_1d + 1.1 * rang / 2
    l4 = close_1d - 1.1 * rang / 2
    h3 = close_1d + 1.1 * rang / 4
    l3 = close_1d - 1.1 * rang / 4
    h2 = close_1d + 1.1 * rang / 6
    l2 = close_1d - 1.1 * rang / 6
    h1 = close_1d + 1.1 * rang / 12
    l1 = close_1d - 1.1 * rang / 12
    
    # Align Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 1week EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 1week close above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1week close below EMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Mean reversion from Camarilla L3/H3 with volume
        # Long when price touches L3 with bullish bias and volume
        # Short when price touches H3 with bearish bias and volume
        # Use close price to avoid wick false signals
        long_entry = bull_bias and (close[i] <= l3_aligned[i]) and volume_confirmed
        short_entry = bear_bias and (close[i] >= h3_aligned[i]) and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

#!/usr/bin/env python3
"""
Experiment #7948: 12-hour Camarilla pivot with 1-week trend filter and volume confirmation.
Hypothesis: Price reacting from weekly Camarilla pivot levels (L3/H3) with volume >1.5x 24-period MA 
and aligned 1week trend (price above/below 1week EMA50) captures mean-reversion in ranging markets 
and continuation in trending markets. Weekly timeframe provides strong trend context to avoid 
counter-trend trades. Target: 50-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7948_12h_camarilla1w_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # daily OHLC for pivot calculation
VOLUME_MA_PERIOD = 24  # 24 * 12h = 12 days
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
EMA_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1week EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Price relative to EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_1w > ema_1w, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1w, price_vs_ema)
    
    # Calculate daily OHLC for Camarilla pivot (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
    # H2 = C + 1.1*(H-L)/6, L2 = C - 1.1*(H-L)/6
    # H1 = C + 1.1*(H-L)/12, L1 = C - 1.1*(H-L)/12
    rang = high_1d - low_1d
    h4 = close_1d + 1.1 * rang / 2
    l4 = close_1d - 1.1 * rang / 2
    h3 = close_1d + 1.1 * rang / 4
    l3 = close_1d - 1.1 * rang / 4
    h2 = close_1d + 1.1 * rang / 6
    l2 = close_1d - 1.1 * rang / 6
    h1 = close_1d + 1.1 * rang / 12
    l1 = close_1d - 1.1 * rang / 12
    
    # Align Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 1week EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 1week close above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1week close below EMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Mean reversion from Camarilla L3/H3 with volume
        # Long when price touches L3 with bullish bias and volume
        # Short when price touches H3 with bearish bias and volume
        # Use close price to avoid wick false signals
        long_entry = bull_bias and (close[i] <= l3_aligned[i]) and volume_confirmed
        short_entry = bear_bias and (close[i] >= h3_aligned[i]) and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals