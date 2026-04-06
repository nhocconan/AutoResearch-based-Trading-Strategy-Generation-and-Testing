#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian Breakout with Daily EMA Trend Filter and Volume Confirmation.
# Uses 20-period Donchian channels on 12h for breakout entries.
# Trend filter: 1d EMA50 - only trade long when above, short when below.
# Volume confirmation: current volume > 1.5x 20-period average.
# Exit on opposite Donchian channel touch or ATR-based stoploss.
# Works in bull/bear markets via trend filter and breakout logic.
# Target: 50-150 trades over 4 years (12-37/year).

name = "12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
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
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = close_1d[49]
        for i in range(50, len(close_1d)):
            ema_50[i] = close_1d[i] * 0.04 + ema_50[i-1] * 0.96
    
    # Align EMA50 to 12h timeframe (shifted by 1 daily bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 20-period Donchian channels on 12h
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(19, n):
        upper[i] = np.max(high[i-19:i+1])
        lower[i] = np.min(low[i-19:i+1])
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # ATR(14) approximation for stoploss
    atr = np.full(n, np.nan)
    for i in range(1, n):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        if i == 1:
            atr[i] = tr
        else:
            atr[i] = 0.93 * atr[i-1] + 0.07 * tr  # Wilder's smoothing
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price touches lower Donchian or stoploss
            if (close[i] <= lower[i] or 
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price touches upper Donchian or stoploss
            if (close[i] >= upper[i] or 
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long breakout above upper Donchian with uptrend
                if (close[i] > upper[i] and close[i-1] <= upper[i] and 
                    close[i] > ema_50_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown below lower Donchian with downtrend
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

# Hypothesis: 12-hour 3-period RSI Extreme with Daily Trend Filter and Volume Confirmation.
# Uses 3-period RSI on 12h for extreme overbought/oversold signals.
# Trend filter: 1d EMA200 - only trade long when above, short when below.
# Volume confirmation: current volume > 2.0x 20-period average to filter noise.
# Exit on RSI returning to neutral zone (40-60) or ATR-based stoploss.
# Works in bull/bear markets via trend filter and mean reversion at extremes.
# Target: 50-150 trades over 4 years (12-37/year).

name = "12h_rsi3_extreme_trend_vol_v1"
timeframe = "12h"
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
    
    # Daily EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        # Calculate initial SMA for EMA seed
        sma = np.mean(close_1d[:200])
        ema_200[199] = sma
        alpha = 2.0 / (200 + 1)
        for i in range(200, len(close_1d)):
            ema_200[i] = close_1d[i] * alpha + ema_200[i-1] * (1 - alpha)
    
    # Align EMA200 to 12h timeframe (shifted by 1 daily bar)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # 3-period RSI on 12h
    rsi = np.full(n, np.nan)
    if n >= 14:
        # Calculate price changes
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing for first 14 periods
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # ATR(14) approximation for stoploss
    atr = np.full(n, np.nan)
    for i in range(1, n):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        if i == 1:
            atr[i] = tr
        else:
            atr[i] = 0.93 * atr[i-1] + 0.07 * tr  # Wilder's smoothing
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI returns to neutral (>=40) or stoploss
            if (rsi[i] >= 40 or 
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: RSI returns to neutral (<=60) or stoploss
            if (rsi[i] <= 60 or 
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long when RSI oversold (<15) in uptrend
                if (rsi[i] < 15 and 
                    close[i] > ema_200_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short when RSI overbought (>85) in downtrend
                elif (rsi[i] > 85 and 
                      close[i] < ema_200_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Kaufman Adaptive Moving Average (KAMA) Trend with Daily Volatility Filter and Volume Confirmation.
# Uses KAMA(10,2,30) on 12h for adaptive trend direction.
# Volatility filter: 1d ATR(14) percentile rank > 0.7 - only trade in high volatility regimes.
# Volume confirmation: current volume > 1.8x 20-period average.
# Exit on KAMA crossover or ATR-based stoploss.
# Works in bull/bear markets via volatility regime filter.
# Target: 50-150 trades over 4 years (12-37/year).

name = "12h_kama_trend_vol_filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily ATR(14) for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14)
    tr_1d = np.full(len(close_1d), np.nan)
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        tr = max(high_1d[i] - low_1d[i], 
                abs(high_1d[i] - close_1d[i-1]), 
                abs(low_1d[i] - close_1d[i-1]))
        tr_1d[i] = tr
        if i == 1:
            atr_1d[i] = tr
        else:
            atr_1d[i] = 0.93 * atr_1d[i-1] + 0.07 * tr  # Wilder's smoothing
    
    # Calculate ATR percentile rank (50-period lookback)
    atr_percentile = np.full(len(close_1d), np.nan)
    for i in range(49, len(close_1d)):
        window = atr_1d[i-49:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            rank = np.sum(valid <= atr_1d[i]) / len(valid)
            atr_percentile[i] = rank
    
    # Align ATR percentile to 12h timeframe (shifted by 1 daily bar)
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # KAMA(10,2,30) on 12h
    kama = np.full(n, np.nan)
    if n >= 30:
        # Efficiency Ratio
        er = np.full(n, np.nan)
        for i in range(9, n):
            direction = abs(close[i] - close[i-9])
            volatility = np.sum(np.abs(np.diff(close[i-9:i+1])))
            if volatility != 0:
                er[i] = direction / volatility
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = np.full(n, np.nan)
        for i in range(9, n):
            fastest = 2 / (2 + 1)   # EMA(2)
            slowest = 2 / (30 + 1)  # EMA(30)
            sc[i] = (er[i] * (fastest - slowest) + slowest) ** 2
        
        # Calculate KAMA
        kama[9] = close[9]
        for i in range(10, n):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # ATR(14) approximation for stoploss
    atr = np.full(n, np.nan)
    for i in range(1, n):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        if i == 1:
            atr[i] = tr
        else:
            atr[i] = 0.93 * atr[i-1] + 0.07 * tr  # Wilder's smoothing
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if data not available
        if (np.isnan(atr_percentile_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume and volatility conditions
        volume_filter = volume[i] > vol_ma[i] * 1.8
        volatility_filter = atr_percentile_aligned[i] > 0.7
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price crosses below KAMA or stoploss
            if (close[i] < kama[i] or 
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above KAMA or stoploss
            if (close[i] > kama[i] or 
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and volatility confirmation
            if volume_filter and volatility_filter:
                # Long when price crosses above KAMA
                if (close[i] > kama[i] and close[i-1] <= kama[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short when price crosses below KAMA
                elif (close[i] < kama[i] and close[i-1] >= kama[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams %R with Daily EMA Filter and Volume Confirmation.
# Uses Williams %R(14) on 12h for overbought/oversold signals.
# Trend filter: 1d EMA50 - only trade long when above, short when below.
# Volume confirmation: current volume > 2.0x 20-period average.
# Exit on Williams %R returning to neutral range (-20 to -80) or ATR stoploss.
# Works in bull/bear markets via trend filter and mean reversion at extremes.
# Target: 50-150 trades over 4 years (12-37/year).

name = "12h_williamsr_ema_vol_v1"
timeframe = "12h"
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
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = close_1d[49]
        alpha = 2.0 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50[i] = close_1d[i] * alpha + ema_50[i-1] * (1 - alpha)
    
    # Align EMA50 to 12h timeframe (shifted by 1 daily bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Williams %R(14) on 12h
    willr = np.full(n, np.nan)
    if n >= 14:
        for i in range(13, n):
            highest_high = np.max(high[i-13:i+1])
            lowest_low = np.min(low[i-13:i+1])
            if highest_high != lowest_low:
                willr[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
            else:
                willr[i] = -50  # neutral when no range
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # ATR(14) approximation for stoploss
    atr = np.full(n, np.nan)
    for i in range(1, n):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        if i == 1:
            atr[i] = tr
        else:
            atr[i] = 0.93 * atr[i-1] + 0.07 * tr  # Wilder's smoothing
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(willr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Williams %R returns to neutral (>= -20) or stoploss
            if (willr[i] >= -20 or 
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R returns to neutral (<= -80) or stoploss
            if (willr[i] <= -80 or 
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long when Williams %R oversold (< -80) in uptrend
                if (willr[i] < -80 and 
                    close[i] > ema_50_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short when Williams %R overbought (> -20) in downtrend
                elif (willr[i] > -20 and 
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

# Hypothesis: 12-hour Elder Ray Index with Daily Trend Filter and Volume Confirmation.
# Uses Elder Ray Index (Bull Power and Bear Power) on 12h for trend strength.
# Trend filter: 1d EMA100 - only trade long when above, short when below.
# Volume confirmation: current volume > 1.5x 20-period average.
# Exit on Elder Ray crossover or ATR-based stoploss.
# Works in bull/bear markets via trend filter and power signals.
# Target: 50-150 trades over 4 years (12-37/year).

name = "12h_elder_ray_trend_vol_v1"
timeframe = "12h"
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
    
    # Daily EMA100 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_100 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 100:
        ema_100[99] = close_1d[99]
        alpha = 2.0 / (100 + 1)
        for i in range(100, len(close_1d)):
            ema_100[i] = close_1d[i] * alpha + ema_100[i-1] * (1 - alpha)
    
    # Align EMA100 to 12h timeframe (shifted by 1 daily bar)
    ema_100_aligned = align_htf_to_ltf(prices, df_1d, ema_100)
    
    # Elder Ray Index components on 12h
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_13 = np.full(n, np.nan)
    if n >= 13:
        ema_13[12] = close[12]
        alpha = 2.0 / (13 + 1)
        for i in range(13, n):
            ema_13[i] = close[i] * alpha + ema_13[i-1] * (1 - alpha)
    
    bull_power = np.full(n, np.nan)
    bear_power = np.full(n, np.nan)
    for i in range(13, n):
        bull_power[i] = high[i] - ema_13[i]
        bear_power[i] = low[i] - ema_13[i]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # ATR(14) approximation for stoploss
    atr = np.full(n, np.nan)
    for i in range(1, n):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        if i == 1:
            atr[i] = tr
        else:
            atr[i] = 0.93 * atr[i-1] + 0.07 * tr  # Wilder's smoothing
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema_100_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Bear Power becomes positive (bulls losing control) or stoploss
            if (bear_power[i] > 0 or 
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bull Power becomes negative (bears losing control) or stoploss
            if (bull_power[i] < 0 or 
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long when Bull Power > 0 and Bear Power < 0 (strong uptrend)
                if (bull_power[i] > 0 and bear_power[i] < 0 and 
                    close[i] > ema_100_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short when Bull Power < 0 and Bear Power > 0 (strong downtrend)
                elif (bull_power[i] < 0 and bear_power[i] > 0 and 
                      close[i] < ema_100_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Vortex Indicator with Daily ATR Filter and Volume Confirmation.
# Uses Vortex Indicator (VI+ and VI-) on 12h for trend direction.
# Volatility filter: 1d ATR(14) > 50th percentile - only trade in sufficient volatility.
# Volume confirmation: current volume > 2.0x 20-period average.
# Exit on Vortex crossover or ATR-based stoploss.
# Works in bull/bear markets via volatility filter and trend signals.
# Target: 50-150 trades over 4 years (12-37/year).

name = "12h_vortex_indicator_vol_v1"
timeframe = "12h"
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
    
    # Daily ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14)
    tr_1d = np.full(len(close_1d), np.nan)
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        tr = max(high_1d[i] - low_1d[i], 
                abs(high_1d[i] - close_1d[i-1]), 
                abs(low_1d[i] - close_1d[i-1]))
        tr_1d[i] = tr
        if i == 1:
            atr_1d[i] = tr
        else:
            atr_1d[i] = 0.93 * atr_1d[i-1] + 0.07 * tr  # Wilder's smoothing
    
    # Calculate ATR percentile rank (50-period lookback)
    atr_percentile = np.full(len(close_1d), np.nan)
    for i in range(49, len(close_1d)):
        window = atr_1d[i-49:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            rank = np.sum(valid <= atr_1d[i]) / len(valid)
            atr_percentile[i] = rank
    
    # Align ATR percentile to 12h timeframe (shifted by 1 daily bar)
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Vortex Indicator on 12h
    vi_plus = np.full(n, np.nan)
    vi_minus = np.full(n, np.nan)
    if n >= 1:
        vm_plus = np.abs(high - np.roll(low, 1))
        vm_minus = np.abs(low - np.roll(high, 1))
        # Handle first element
        vm_plus[0] = np.abs(high[0] - low[0])
        vm_minus[0] = np.abs(low[0] - high[0])
        
        # Sum over 14 periods
        vi14_plus = np.full(n, np.nan)