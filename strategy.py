#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour 55-period EMA trend + 20-period Bollinger Band squeeze breakout with volume confirmation
# Long when price breaks above upper BB with EMA > previous EMA and volume > 1.5x average
# Short when price breaks below lower BB with EMA < previous EMA and volume > 1.5x average
# Exit when price returns to middle BB or volatility contraction fails
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 100-180 total trades over 4 years (25-45/year)

name = "4h_ema_bb_squeeze_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA(55) for trend direction
    close_s = pd.Series(close)
    ema = close_s.ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Bollinger Bands (20, 2)
    sma = close_s.rolling(window=20, min_periods=20).mean().values
    std = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = sma + 2 * std
    lower_bb = sma - 2 * std
    middle_bb = sma
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(55, n):
        # Skip if required data not available
        if (np.isnan(ema[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(middle_bb[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to middle BB or EMA turns down
            elif close[i] <= middle_bb[i] or ema[i] < ema[i-1]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to middle BB or EMA turns up
            elif close[i] >= middle_bb[i] or ema[i] > ema[i-1]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with volume confirmation
            # Volume spike: > 1.5x average volume
            volume_spike = volume[i] > 1.5 * volume_ma[i]
            
            # Long: price breaks above upper BB with rising EMA and volume spike
            if (close[i] > upper_bb[i] and ema[i] > ema[i-1] and volume_spike):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower BB with falling EMA and volume spike
            elif (close[i] < lower_bb[i] and ema[i] < ema[i-1] and volume_spike):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Bollinger Band squeeze breakout with EMA trend filter and volume confirmation
# Long when price breaks above upper BB with EMA(50) rising and volume > 1.5x average
# Short when price breaks below lower BB with EMA(50) falling and volume > 1.5x average
# Exit when price returns to middle BB or EMA direction changes
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 100-180 total trades over 4 years (25-45/year)

name = "4h_bb_squeeze_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA(50) for trend direction
    close_s = pd.Series(close)
    ema = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Bollinger Bands (20, 2)
    sma = close_s.rolling(window=20, min_periods=20).mean().values
    std = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = sma + 2 * std
    lower_bb = sma - 2 * std
    middle_bb = sma
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(middle_bb[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to middle BB or EMA turns down
            elif close[i] <= middle_bb[i] or ema[i] < ema[i-1]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to middle BB or EMA turns up
            elif close[i] >= middle_bb[i] or ema[i] > ema[i-1]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with volume confirmation
            # Volume spike: > 1.5x average volume
            volume_spike = volume[i] > 1.5 * volume_ma[i]
            
            # Long: price breaks above upper BB with rising EMA and volume spike
            if (close[i] > upper_bb[i] and ema[i] > ema[i-1] and volume_spike):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower BB with falling EMA and volume spike
            elif (close[i] < lower_bb[i] and ema[i] < ema[i-1] and volume_spike):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Bollinger Band squeeze breakout with EMA trend filter and volume confirmation
# Long when price breaks above upper BB with EMA(50) rising and volume > 1.5x average
# Short when price breaks below lower BB with EMA(50) falling and volume > 1.5x average
# Exit when price returns to middle BB or EMA direction changes
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 100-180 total trades over 4 years (25-45/year)

name = "4h_bb_squeeze_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA(50) for trend direction
    close_s = pd.Series(close)
    ema = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Bollinger Bands (20, 2)
    sma = close_s.rolling(window=20, min_periods=20).mean().values
    std = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = sma + 2 * std
    lower_bb = sma - 2 * std
    middle_bb = sma
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(middle_bb[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to middle BB or EMA turns down
            elif close[i] <= middle_bb[i] or ema[i] < ema[i-1]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to middle BB or EMA turns up
            elif close[i] >= middle_bb[i] or ema[i] > ema[i-1]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with volume confirmation
            # Volume spike: > 1.5x average volume
            volume_spike = volume[i] > 1.5 * volume_ma[i]
            
            # Long: price breaks above upper BB with rising EMA and volume spike
            if (close[i] > upper_bb[i] and ema[i] > ema[i-1] and volume_spike):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower BB with falling EMA and volume spike
            elif (close[i] < lower_bb[i] and ema[i] < ema[i-1] and volume_spike):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

---  ## Failed check: <50 trades on train (0 < 50)