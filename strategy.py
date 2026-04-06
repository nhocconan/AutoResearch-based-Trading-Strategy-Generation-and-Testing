#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly breakout with volume confirmation and ATR stoploss
# Enter long when: price breaks above weekly high + volume spike
# Enter short when: price breaks below weekly low + volume spike
# Exit on opposite breakout or ATR stoploss
# Uses weekly structure to capture multi-day trends, targeting 30-100 trades over 4 years

name = "1d_weekly_breakout_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for structure
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Align weekly levels to daily (already shifted by 1 week for completed bars)
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low)
    
    # Volume confirmation: volume > 2.0x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(volume_threshold[i]) or np.isnan(atr[i])):
            if position != 0:
                # Maintain position with stoploss check
                if position == 1 and close[i] < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: opposite breakout or ATR stoploss
        if position == 1:  # long position
            if close[i] < weekly_low_aligned[i] or close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > weekly_high_aligned[i] or close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: weekly breakout + volume spike
            if volume[i] > volume_threshold[i]:
                if close[i] > weekly_high_aligned[i]:
                    # Break above weekly high with volume - bullish breakout
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < weekly_low_aligned[i]:
                    # Break below weekly low with volume - bearish breakout
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly breakout with volume confirmation and ATR stoploss
# Enter long when: price breaks above weekly high + volume spike
# Enter short when: price breaks below weekly low + volume spike
# Exit on opposite breakout or ATR stoploss
# Uses weekly structure to capture multi-day trends, targeting 30-100 trades over 4 years

name = "1d_weekly_breakout_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for structure
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Align weekly levels to daily (already shifted by 1 week for completed bars)
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low)
    
    # Volume confirmation: volume > 2.0x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(volume_threshold[i]) or np.isnan(atr[i])):
            if position != 0:
                # Maintain position with stoploss check
                if position == 1 and close[i] < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: opposite breakout or ATR stoploss
        if position == 1:  # long position
            if close[i] < weekly_low_aligned[i] or close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > weekly_high_aligned[i] or close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: weekly breakout + volume spike
            if volume[i] > volume_threshold[i]:
                if close[i] > weekly_high_aligned[i]:
                    # Break above weekly high with volume - bullish breakout
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < weekly_low_aligned[i]:
                    # Break below weekly low with volume - bearish breakout
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals