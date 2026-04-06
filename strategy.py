#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel during bullish day with volume > 1.3x 20-period average.
# Short when price breaks below lower Donchian channel during bearish day with volume confirmation.
# Uses daily trend filter to avoid counter-trend trades. Donchian channels provide clear breakout points.
# Target: 75-150 total trades over 4 years (19-38/year) to stay within optimal range.

name = "4h_donchian20_1d_trend_vol_v1"
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
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily trend filter: bullish/bearish day based on close vs open
    df_1d = get_htf_data(prices, '1d')
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    daily_bullish = daily_close > daily_open  # True for bullish day
    daily_bearish = daily_close < daily_open   # True for bearish day
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if daily trend data not available
        if np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits
        if position == 1:  # long position
            # Exit: price drops below lower Donchian or daily turn bearish
            if (low[i] <= lower[i] or 
                daily_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above upper Donchian or daily turn bullish
            if (high[i] >= upper[i] or 
                daily_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and daily trend filter
            if volume_filter:
                # Long: break above upper Donchian during bullish day
                if (high[i] > upper[i] and 
                    daily_bullish_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: break below lower Donchian during bearish day
                elif (low[i] < lower[i] and 
                      daily_bearish_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(25) breakout with 1d EMA filter and volume confirmation.
# Long when price breaks above upper Donchian channel with price above 1d EMA(50) and volume > 1.5x 20-period average.
# Short when price breaks below lower Donchian channel with price below 1d EMA(50) and volume > 1.5x 20-period average.
# Uses daily EMA to avoid counter-trend trades. Higher volume threshold reduces trade frequency.
# Target: 75-150 total trades over 4 years (19-38/year) to stay within optimal range.

name = "4h_donchian25_1d_ema_vol_v1"
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
    
    # Donchian channel (25-period) - slightly wider to reduce false breakouts
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=25, min_periods=25).max().values
    lower = low_series.rolling(window=25, min_periods=25).min().values
    
    # Daily EMA filter: trend direction from 1d EMA(50)
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_above_ema = daily_close > daily_ema  # True when price above EMA (bullish)
    daily_below_ema = daily_close < daily_ema   # True when price below EMA (bearish)
    daily_above_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_above_ema)
    daily_below_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_below_ema)
    
    # Volume filter: current volume > 1.5x 20-period average (higher threshold to reduce trades)
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(25, n):
        # Skip if daily EMA data not available
        if np.isnan(daily_above_ema_aligned[i]) or np.isnan(daily_below_ema_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition with higher threshold
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: price drops below lower Donchian or price crosses below 1d EMA
            if (low[i] <= lower[i] or 
                not daily_above_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above upper Donchian or price crosses above 1d EMA
            if (high[i] >= upper[i] or 
                not daily_below_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and daily EMA filter
            if volume_filter:
                # Long: break above upper Donchian with price above 1d EMA
                if (high[i] > upper[i] and 
                    daily_above_ema_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: break below lower Donchian with price below 1d EMA
                elif (low[i] < lower[i] and 
                      daily_below_ema_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX filter and volume confirmation.
# Long when price breaks above upper Donchian channel with 1d ADX > 25 (trending market) and volume > 1.4x 20-period average.
# Short when price breaks below lower Donchian channel with 1d ADX > 25 and volume confirmation.
# Uses ADX to filter for trending markets only, avoiding ranging conditions that cause whipsaws.
# Higher volume threshold and ADX filter reduce trade frequency to optimal range.
# Target: 75-150 total trades over 4 years (19-38/year).

name = "4h_donchian20_1d_adx_vol_v1"
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
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily ADX filter: only trade in trending markets (ADX > 25)
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate ADX components
    # +DM and -DM
    high_diff = np.diff(daily_high, prepend=daily_high[0])
    low_diff = np.diff(daily_low, prepend=daily_low[0])
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # True Range
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.roll(daily_close, 1))
    tr3 = np.abs(daily_low - np.roll(daily_close, 1))
    tr1[0] = daily_high[0] - daily_low[0]  # first period
    tr2[0] = np.abs(daily_high[0] - daily_close[0])
    tr3[0] = np.abs(daily_low[0] - daily_close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        for i in range(len(data)):
            if np.isnan(result[i-1]) if i > 0 else True:
                result[i] = data[i]
            else:
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    period = 14
    atr = wilders_smooth(tr, period)
    plus_di = 100 * wilders_smooth(plus_dm, period) / atr
    minus_di = 100 * wilders_smooth(minus_dm, period) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, period)
    
    # ADX > 25 indicates trending market
    daily_trending = adx > 25.0
    daily_trending_aligned = align_htf_to_ltf(prices, df_1d, daily_trending)
    
    # Volume filter: current volume > 1.4x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if daily ADX data not available
        if np.isnan(daily_trending_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.4
        
        # Check exits
        if position == 1:  # long position
            # Exit: price drops below lower Donchian or market becomes non-trending
            if (low[i] <= lower[i] or 
                not daily_trending_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above upper Donchian or market becomes non-trending
            if (high[i] >= upper[i] or 
                not daily_trending_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and ADX filter
            if volume_filter and daily_trending_aligned[i]:
                # Long: break above upper Donchian in trending market
                if high[i] > upper[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: break below lower Donchian in trending market
                elif low[i] < lower[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d RSI filter and volume confirmation.
# Long when price breaks above upper Donchian channel with 1d RSI < 50 (avoiding overbought) and volume > 1.3x 20-period average.
# Short when price breaks below lower Donchian channel with 1d RSI > 50 (avoiding oversold) and volume confirmation.
# Uses RSI to avoid extreme conditions that often reverse. Volume filter ensures momentum confirmation.
# Target: 75-150 total trades over 4 years (19-38/year).

name = "4h_donchian20_1d_rsi_vol_v1"
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
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily RSI filter: avoid extremes
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    # Calculate RSI (14-period)
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        for i in range(len(data)):
            if np.isnan(result[i-1]) if i > 0 else True:
                result[i] = data[i]
            else:
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    period = 14
    avg_gain = wilders_smooth(gain, period)
    avg_loss = wilders_smooth(loss, period)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # RSI < 50 for long bias (not overbought), RSI > 50 for short bias (not oversold)
    daily_rsi_low = rsi < 50.0   # Favors longs (not overbought)
    daily_rsi_high = rsi > 50.0  # Favors shorts (not oversold)
    daily_rsi_low_aligned = align_htf_to_ltf(prices, df_1d, daily_rsi_low)
    daily_rsi_high_aligned = align_htf_to_ltf(prices, df_1d, daily_rsi_high)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if daily RSI data not available
        if np.isnan(daily_rsi_low_aligned[i]) or np.isnan(daily_rsi_high_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits
        if position == 1:  # long position
            # Exit: price drops below lower Donchian or RSI becomes too high (>60)
            if (low[i] <= lower[i] or 
                not daily_rsi_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above upper Donchian or RSI becomes too low (<40)
            if (high[i] >= upper[i] or 
                not daily_rsi_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and RSI filter
            if volume_filter:
                # Long: break above upper Donchian with RSI < 50 (not overbought)
                if (high[i] > upper[i] and 
                    daily_rsi_low_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: break below lower Donchian with RSI > 50 (not oversold)
                elif (low[i] < lower[i] and 
                      daily_rsi_high_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume imbalance filter.
# Long when price breaks above upper Donchian channel with 1d volume > 1.5x previous day and 4h volume > 1.4x 20-period average.
# Short when price breaks below lower Donchian channel with 1d volume < 0.7x previous day and 4h volume confirmation.
# Uses daily volume imbalance to detect institutional interest. Lower frequency due to dual volume confirmation.
# Target: 50-100 total trades over 4 years (12-25/year) to minimize fee drag.

name = "4h_donchian20_1d_volimb_v1"
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
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily volume imbalance: today's volume vs previous day
    df_1d = get_htf_data(prices, '1d')
    daily_volume = df_1d['volume'].values
    daily_vol_prev = np.roll(daily_volume, 1)  # previous day's volume
    daily_vol_prev[0] = daily_volume[0]  # first period
    
    # Volume imbalance conditions
    daily_vol_high = daily_volume > (daily_vol_prev * 1.5)  # volume spike up
    daily_vol_low = daily_volume < (daily_vol_prev * 0.7)   # volume drop
    daily_vol_high_aligned = align_htf_to_ltf(prices, df_1d, daily_vol_high)
    daily_vol_low_aligned = align_htf_to_ltf(prices, df_1d, daily_vol_low)
    
    # 4h volume filter: current volume > 1.4x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if daily volume data not available
        if np.isnan(daily_vol_high_aligned[i]) or np.isnan(daily_vol_low_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # 4h volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.4
        
        # Check exits
        if position == 1:  # long position
            # Exit: price drops below lower Donchian or volume imbalance reverses
            if (low[i] <= lower[i] or 
                not daily_vol_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above upper Donchian or volume imbalance reverses
            if (high[i] >= upper[i] or 
                not daily_vol_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and daily imbalance
            if volume_filter:
                # Long: break above upper Donchian with daily volume spike
                if (high[i] > upper[i] and 
                    daily_vol_high_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: break below lower Donchian with daily volume drop
                elif (low[i] < lower[i] and 
                      daily_vol_low_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d price action filter (higher highs/lows).
# Long when price breaks above upper Donchian channel with 1d making higher high and higher low.
# Short when price breaks below lower Donchian channel with 1d making lower high and lower low.
# Uses daily price structure to confirm trend integrity. Reduces counter-trend trades.
# Target: 60-120 total trades over 4 years (15-30/year).

name = "4h_donchian20_1d_priceaction_v1"
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
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily price action: higher highs/lows vs lower highs/lows
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Higher High and Higher Low (HHHL) - bullish structure
    daily_hh = daily_high > np.roll(daily_high, 1)  # higher high than previous day
    daily_hl = daily_low > np.roll(daily_low, 1)    # higher low than previous day
    daily_hh_hl = daily_hh & daily_hl               # both conditions
    
    # Lower High and Lower Low (LHLL) - bearish structure
    daily_lh = daily_high < np.roll(daily_high, 1)  # lower high than previous day
    daily_ll = daily_low < np.roll(daily_low, 1)    # lower low than previous day
    daily_lh_ll = daily_lh & daily_ll               # both conditions
    
    # Handle first day
    daily_hh[0] = False
    daily_hl[0] = False
    daily_lh[0] = False
    daily_ll[0] = False
    daily_hh_hl[0] = False
    daily_lh_ll[0] = False
    
    daily_hh_hl_aligned = align_htf_to_ltf(prices, df_1d, daily_hh_hl)
    daily_lh_ll_aligned = align_htf_to_ltf(prices, df_1d, daily_lh_ll)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if daily price action data not available
        if np.isnan(daily_hh_hl_aligned[i]) or np.isnan(daily_lh_ll_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits
        if position == 1:  # long position
            # Exit: price drops below lower Donchian or price structure breaks
            if (low[i] <= lower[i] or 
                not daily_hh_hl_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above upper Donchian or price structure breaks
            if (high[i] >= upper[i] or 
                not daily_lh_ll_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and price action filter
            if volume_filter:
                # Long: break above upper Donchian with HHHL structure
                if (high[i] > upper[i] and 
                    daily_hh_hl_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: break below lower Donchian with LHLL structure
                elif (low[i] < lower[i] and 
                      daily_lh_ll_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volatility filter (low volatility breakout).
# Long when price breaks above upper Donchian channel during low volatility day (1d ATR < 20-period MA) and volume > 1.3x.
# Short when price breaks below lower Donchian channel during low volatility day and volume confirmation.
# Uses low volatility to catch breakouts before expansion. Volume confirms follow-through.
# Target: 70-140 total trades over 4 years (17-35/year).

name = "4h_donchian20_1d_volfilter_v1"
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
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily volatility filter: low volatility environment
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate True Range and ATR (14-period)
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.roll(daily_close, 1))
    tr3 = np.abs(daily_low - np.roll(daily_close, 1))
    tr1[0] = daily_high[0] - daily_low[0]
    tr2[0] = np.abs(daily_high[0] - daily_close[0])
    tr3[0] = np.abs(daily_low[0] - daily_close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing for ATR
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        for i in range(len(data)):
            if np.isnan(result[i-1]) if i > 0 else True:
                result[i] = data[i]
            else:
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr = wilders_smooth(tr, 14)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # Low volatility: current ATR below 20-period average
    daily_low_vol = atr < atr_ma
    daily_low_vol_aligned = align_htf_to_ltf(prices, df_1d, daily_low_vol)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if daily volatility data not available
        if np.isnan(daily_low_vol_aligned[i]):
            if position != 0:
                signals[i] = position *