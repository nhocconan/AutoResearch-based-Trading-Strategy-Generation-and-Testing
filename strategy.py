#!/usr/bin/env python3
"""
Experiment #8211: 6-hour Williams Alligator with 1-day trend filter and volume confirmation.
Hypothesis: The Williams Alligator (Jaws, Teeth, Lips) identifies trending vs ranging markets on 6h.
In trending markets (JAW > TEETH > LIPS for up, JAW < TEETH < LIPS for down), we trade in the direction
of the 1-day trend confirmed by price > EMA50 (bull) or price < EMA50 (bear), with volume > 1.5x 20-period MA.
This avoids whipsaw in ranging markets and captures sustained moves in both bull and bear regimes.
Targeting 100-200 total trades over 4 years for optimal balance of signal quality and cost.
"""

from mtf_data import get_ath_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8211_6h_williams_alligator_1d_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13
ALLIGATOR_TEETH_PERIOD = 8
ALLIGATOR_LIPS_PERIOD = 5
VOLUME_MA_PERIOD = 20
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Price relative to EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_1d > ema_1d, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: SMMA (Smoothed Moving Average)
    def smma(series, period):
        """Smoothed Moving Average - similar to Wilder's smoothing"""
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
        # Apply smoothing: SMMA(t) = (SMMA(t-1) * (period-1) + price(t)) / period
        smma_vals = np.full_like(series, np.nan, dtype=float)
        if len(series) >= period:
            smma_vals[period-1] = sma[period-1]  # First value is SMA
            for i in range(period, len(series)):
                if not np.isnan(smma_vals[i-1]):
                    smma_vals[i] = (smma_vals[i-1] * (period-1) + series[i]) / period
                else:
                    smma_vals[i] = np.nan
        return smma_vals
    
    jaw = smma(high, ALLIGATOR_JAW_PERIOD)   # Smoothed High (Blue line)
    teeth = smma((high + low) / 2, ALLIGATOR_TEETH_PERIOD)  # Smoothed Mid (Red line)
    lips = smma(low, ALLIGATOR_LIPS_PERIOD)   # Smoothed Low (Green line)
    
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
    start = max(ALLIGATOR_JAW_PERIOD, ALLIGATOR_TEETH_PERIOD, ALLIGATOR_LIPS_PERIOD, 
                VOLUME_MA_PERIOD, ATR_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]):
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
        
        # Alligator alignment - check for proper alignment (no crossing)
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Skip if any Alligator line is not available
        if np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Bullish alignment: Lips > Teeth > Jaw (Green > Red > Blue)
        bullish_alignment = (lips_val > teeth_val) and (teeth_val > jaw_val)
        # Bearish alignment: Jaw > Teeth > Lips (Blue > Red > Green)
        bearish_alignment = (jaw_val > teeth_val) and (teeth_val > lips_val)
        
        # Determine market bias from 1d EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d close above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d close below EMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bullish_alignment and bull_bias and volume_confirmed
        short_entry = bearish_alignment and bear_bias and volume_confirmed
        
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

</think>
#!/usr/bin/env python3
"""
Experiment #8211: 6-hour Williams Alligator with 1-day trend filter and volume confirmation.
Hypothesis: The Williams Alligator (Jaws, Teeth, Lips) identifies trending vs ranging markets on 6h.
In trending markets (JAW > TEETH > LIPS for up, JAW < TEETH < LIPS for down), we trade in the direction
of the 1-day trend confirmed by price > EMA50 (bull) or price < EMA50 (bear), with volume > 1.5x 20-period MA.
This avoids whipsaw in ranging markets and captures sustained moves in both bull and bear regimes.
Targeting 100-200 total trades over 4 years for optimal balance of signal quality and cost.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8211_6h_williams_alligator_1d_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13
ALLIGATOR_TEETH_PERIOD = 8
ALLIGATOR_LIPS_PERIOD = 5
VOLUME_MA_PERIOD = 20
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Price relative to EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_1d > ema_1d, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: SMMA (Smoothed Moving Average)
    def smma(series, period):
        """Smoothed Moving Average - similar to Wilder's smoothing"""
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
        # Apply smoothing: SMMA(t) = (SMMA(t-1) * (period-1) + price(t)) / period
        smma_vals = np.full_like(series, np.nan, dtype=float)
        if len(series) >= period:
            smma_vals[period-1] = sma[period-1]  # First value is SMA
            for i in range(period, len(series)):
                if not np.isnan(smma_vals[i-1]):
                    smma_vals[i] = (smma_vals[i-1] * (period-1) + series[i]) / period
                else:
                    smma_vals[i] = np.nan
        return smma_vals
    
    jaw = smma(high, ALLIGATOR_JAW_PERIOD)   # Smoothed High (Blue line)
    teeth = smma((high + low) / 2, ALLIGATOR_TEETH_PERIOD)  # Smoothed Mid (Red line)
    lips = smma(low, ALLIGATOR_LIPS_PERIOD)   # Smoothed Low (Green line)
    
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
    start = max(ALLIGATOR_JAW_PERIOD, ALLIGATOR_TEETH_PERIOD, ALLIGATOR_LIPS_PERIOD, 
                VOLUME_MA_PERIOD, ATR_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]):
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
        
        # Alligator alignment - check for proper alignment (no crossing)
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Skip if any Alligator line is not available
        if np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Bullish alignment: Lips > Teeth > Jaw (Green > Red > Blue)
        bullish_alignment = (lips_val > teeth_val) and (teeth_val > jaw_val)
        # Bearish alignment: Jaw > Teeth > Lips (Blue > Red > Green)
        bearish_alignment = (jaw_val > teeth_val) and (teeth_val > lips_val)
        
        # Determine market bias from 1d EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d close above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d close below EMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bullish_alignment and bull_bias and volume_confirmed
        short_entry = bearish_alignment and bear_bias and volume_confirmed
        
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