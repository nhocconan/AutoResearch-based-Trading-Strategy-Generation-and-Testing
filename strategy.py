#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian Breakout with Daily EMA Trend Filter and Volume Confirmation.
# Uses 20-period Donchian channels on 4h timeframe to capture breakouts.
# Trend filter: 50-period EMA on daily timeframe to avoid counter-trend trades.
# Volume filter: current volume > 1.5x 20-period average to ensure quality breakouts.
# Works in bull/bear markets by aligning with higher timeframe trend.
# Target: 75-200 trades over 4 years (19-50/year).

name = "4h_donchian20_1d_ema50_vol_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(close_1d), np.nan)
    for i in range(49, len(close_1d)):
        if i == 49:
            ema_50[i] = np.mean(close_1d[0:50])
        else:
            ema_50[i] = close_1d[i] * 0.0377 + ema_50[i-1] * 0.9623
    
    # Align daily EMA to 4h timeframe (shifted by 1 daily bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Donchian channels on 4h timeframe
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(19, n):
        upper[i] = np.max(high[i-19:i+1])
        lower[i] = np.min(low[i-19:i+1])
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (close[i] < lower[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (close[i] > upper[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume[i] > vol_ma[i] * 1.5:
                # Long breakout: price closes above upper channel with uptrend
                if (close[i] > upper[i] and close[i-1] <= upper[i] and 
                    close[i] > ema_50_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price closes below lower channel with downtrend
                elif (close[i] < lower[i] and close[i-1] >= lower[i] and 
                      close[i] < ema_50_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian Breakout with Daily EMA Trend Filter and Volume Confirmation.
# Uses 20-period Donchian channels on 4h timeframe to capture breakouts.
# Trend filter: 50-period EMA on daily timeframe to avoid counter-trend trades.
# Volume filter: current volume > 1.5x 20-period average to ensure quality breakouts.
# Works in bull/bear markets by aligning with higher timeframe trend.
# Target: 75-200 trades over 4 years (19-50/year).

name = "4h_donchian20_1d_ema50_vol_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(close_1d), np.nan)
    for i in range(49, len(close_1d)):
        if i == 49:
            ema_50[i] = np.mean(close_1d[0:50])
        else:
            ema_50[i] = close_1d[i] * 0.0377 + ema_50[i-1] * 0.9623
    
    # Align daily EMA to 4h timeframe (shifted by 1 daily bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Donchian channels on 4h timeframe
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(19, n):
        upper[i] = np.max(high[i-19:i+1])
        lower[i] = np.min(low[i-19:i+1])
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (close[i] < lower[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (close[i] > upper[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume[i] > vol_ma[i] * 1.5:
                # Long breakout: price closes above upper channel with uptrend
                if (close[i] > upper[i] and close[i-1] <= upper[i] and 
                    close[i] > ema_50_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price closes below lower channel with downtrend
                elif (close[i] < lower[i] and close[i-1] >= lower[i] and 
                      close[i] < ema_50_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian Breakout with Daily EMA Trend Filter and Volume Confirmation.
# Uses 20-period Donchian channels on 4h timeframe to capture breakouts.
# Trend filter: 50-period EMA on daily timeframe to avoid counter-trend trades.
# Volume filter: current volume > 1.5x 20-period average to ensure quality breakouts.
# Works in bull/bear markets by aligning with higher timeframe trend.
# Target: 75-200 trades over 4 years (19-50/year).

name = "4h_donchian20_1d_ema50_vol_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(close_1d), np.nan)
    for i in range(49, len(close_1d)):
        if i == 49:
            ema_50[i] = np.mean(close_1d[0:50])
        else:
            ema_50[i] = close_1d[i] * 0.0377 + ema_50[i-1] * 0.9623
    
    # Align daily EMA to 4h timeframe (shifted by 1 daily bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Donchian channels on 4h timeframe
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(19, n):
        upper[i] = np.max(high[i-19:i+1])
        lower[i] = np.min(low[i-19:i+1])
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (close[i] < lower[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (close[i] > upper[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume[i] > vol_ma[i] * 1.5:
                # Long breakout: price closes above upper channel with uptrend
                if (close[i] > upper[i] and close[i-1] <= upper[i] and 
                    close[i] > ema_50_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price closes below lower channel with downtrend
                elif (close[i] < lower[i] and close[i-1] >= lower[i] and 
                      close[i] < ema_50_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

---  End of script ---