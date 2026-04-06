#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h volume confirmation and 1d ATR filter
# Long when price breaks above Donchian high (20) + volume > 1.5x 20-period average + ATR(14) < 0.5 * ATR(50)
# Short when price breaks below Donchian low (20) + volume > 1.5x 20-period average + ATR(14) < 0.5 * ATR(50)
# Exit when price crosses Donchian midline (10-period average of high/low) or ATR condition fails
# Uses volume to confirm breakouts and ATR filter to avoid whipsaws in high volatility
# Targets 50-150 trades over 4 years by requiring multiple confirmations

name = "6h_donchian20_vol_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    # ATR calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean()
    
    # ATR filter: short-term ATR < 50% of long-term ATR (low volatility environment)
    atr_long = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean()
    atr_condition = atr.values < (0.5 * atr_long.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or \
           np.isnan(volume_threshold[i]) or np.isnan(atr_condition[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses midline or volatility increases
        if position == 1:  # long position
            if close[i] <= donch_mid[i] or not atr_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donch_mid[i] or not atr_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with volume confirmation and low volatility
            # Long: break above Donchian high + volume confirmation + low volatility
            if close[i] > donch_high[i-1] and volume[i] > volume_threshold[i] and atr_condition[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + volume confirmation + low volatility
            elif close[i] < donch_low[i-1] and volume[i] > volume_threshold[i] and atr_condition[i]:
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h volume confirmation and 1d ATR filter
# Long when price breaks above Donchian high (20) + volume > 1.5x 20-period average + ATR(14) < 0.5 * ATR(50)
# Short when price breaks below Donchian low (20) + volume > 1.5x 20-period average + ATR(14) < 0.5 * ATR(50)
# Exit when price crosses Donchian midline (10-period average of high/low) or ATR condition fails
# Uses volume to confirm breakouts and ATR filter to avoid whipsaws in high volatility
# Targets 50-150 trades over 4 years by requiring multiple confirmations

name = "6h_donchian20_vol_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    # ATR calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean()
    
    # ATR filter: short-term ATR < 50% of long-term ATR (low volatility environment)
    atr_long = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean()
    atr_condition = atr.values < (0.5 * atr_long.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or \
           np.isnan(volume_threshold[i]) or np.isnan(atr_condition[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses midline or volatility increases
        if position == 1:  # long position
            if close[i] <= donch_mid[i] or not atr_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donch_mid[i] or not atr_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with volume confirmation and low volatility
            # Long: break above Donchian high + volume confirmation + low volatility
            if close[i] > donch_high[i-1] and volume[i] > volume_threshold[i] and atr_condition[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + volume confirmation + low volatility
            elif close[i] < donch_low[i-1] and volume[i] > volume_threshold[i] and atr_condition[i]:
                signals[i] = -0.25
                position = -1
    
    return signals