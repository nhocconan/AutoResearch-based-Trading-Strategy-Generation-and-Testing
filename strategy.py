#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA trend filter and volume confirmation.
# Goes long when price breaks above 20-period high with 1w EMA uptrend and volume > average.
# Goes short when price breaks below 20-period low with 1w EMA downtrend and volume > average.
# Uses ATR-based stoploss to limit downside. Designed to work in both bull and bear markets
# by following the trend on higher timeframe (1w) while capturing breakouts on 12h.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.

name = "12h_donchian20_1w_ema_vol_v1"
timeframe = "12h"
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
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w close
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Donchian channels (20-period) on 12h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume moving average for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian low or 1w EMA turns down
            elif close[i] < low_min[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian high or 1w EMA turns up
            elif close[i] > high_max[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume filter
            if vol_filter:
                # Long entry: price breaks above Donchian high with 1w EMA uptrend
                if close[i] > high_max[i] and close[i] > ema_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short entry: price breaks below Donchian low with 1w EMA downtrend
                elif close[i] < low_min[i] and close[i] < ema_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot-based mean reversion with 1d trend filter and volume confirmation.
# Uses 1d close to determine trend (above/below EMA50). In uptrend, look for long at L3 support;
# in downtrend, look for short at H3 resistance. Uses volume spike (>1.5x average) for confirmation.
# Includes ATR-based stoploss (2x ATR) and time-based exit (10 bars) to limit losses.
# Designed to work in ranging markets with clear intraday structure.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.

name = "12h_camarilla1d_trend_vol_v1"
timeframe = "12h"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close for trend
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels using previous 12h bar's range
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    # We use previous bar's range to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    range_ = prev_high - prev_low
    H3 = prev_close + 1.1 * range_
    L3 = prev_close - 1.1 * range_
    
    # Volume moving average for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_in_trade = 0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(H3[i]) or np.isnan(L3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require volume above 1.5x average
        vol_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_in_trade = 0
            # Time-based exit: max 10 bars
            elif bars_in_trade >= 10:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_in_trade = 0
            else:
                signals[i] = 0.25
                bars_in_trade += 1
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_in_trade = 0
            # Time-based exit: max 10 bars
            elif bars_in_trade >= 10:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_in_trade = 0
            else:
                signals[i] = -0.25
                bars_in_trade += 1
        else:
            # Look for entries with volume filter and trend alignment
            if vol_filter:
                # Determine trend from 1d EMA: above = uptrend, below = downtrend
                uptrend = close[i] > ema_1d_aligned[i]
                downtrend = close[i] < ema_1d_aligned[i]
                
                # Long entry: price at L3 support in uptrend
                if uptrend and close[i] <= L3[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_in_trade = 1
                # Short entry: price at H3 resistance in downtrend
                elif downtrend and close[i] >= H3[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_in_trade = 1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Trix momentum with 1w trend filter and volume confirmation.
# Uses Trix(12) on 12h for momentum, 1w EMA50 for trend filter, and volume spike (>2x average) for confirmation.
# Goes long when Trix crosses above zero with 1w uptrend and volume confirmation.
# Goes short when Trix crosses below zero with 1w downtrend and volume confirmation.
# Includes ATR-based stoploss (2x ATR) and trailing stop (1.5x ATR from peak) to manage risk.
# Designed to capture momentum shifts in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.

name = "12h_trix1w_vol_v1"
timeframe = "12h"
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
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w close for trend
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Trix(12) on 12h close
    # Trix = EMA(EMA(EMA(close, 12), 12), 12) - 1
    ema1 = pd.Series(close).ewm(span=12, min_periods=12, adjust=False).mean()
    ema2 = ema1.ewm(span=12, min_periods=12, adjust=False).mean()
    ema3 = ema2.ewm(span=12, min_periods=12, adjust=False).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100
    trix = trix.fillna(0).values
    
    # Volume moving average for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    peak_price = 0.0  # for trailing stop in long
    trough_price = 0.0  # for trailing stop in short
    
    for i in range(12, n):
        # Skip if required data not available
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(trix[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require volume above 2x average
        vol_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # long position
            # Update peak price
            if close[i] > peak_price:
                peak_price = close[i]
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                peak_price = 0.0
            # Trailing stop: 1.5 * ATR below peak
            elif close[i] < peak_price - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                peak_price = 0.0
            # Exit: Trix crosses below zero or 1w EMA turns down
            elif trix[i] < 0 or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                peak_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Update trough price
            if close[i] < trough_price:
                trough_price = close[i]
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                trough_price = 0.0
            # Trailing stop: 1.5 * ATR above trough
            elif close[i] > trough_price + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                trough_price = 0.0
            # Exit: Trix crosses above zero or 1w EMA turns up
            elif trix[i] > 0 or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                trough_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume filter
            if vol_filter:
                # Long entry: Trix crosses above zero with 1w uptrend
                if trix[i] > 0 and trix[i-1] <= 0 and close[i] > ema_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    peak_price = close[i]
                # Short entry: Trix crosses below zero with 1w downtrend
                elif trix[i] < 0 and trix[i-1] >= 0 and close[i] < ema_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    trough_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA adaptive trend with 1d trend filter and volume confirmation.
# Uses Kaufman's Adaptive Moving Average (KAMA) on 12h for trend, 1d EMA50 for higher timeframe filter,
# and volume spike (>1.5x average) for confirmation. Goes long when price crosses above KAMA
# with 1d uptrend and volume confirmation; short when price crosses below KAMA with 1d downtrend.
# Includes ATR-based stoploss (2x ATR) and time-based exit (8 bars) to limit losses.
# Designed to work in both trending and ranging markets by adapting to market conditions.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.

name = "12h_kama1d_vol_v1"
timeframe = "12h"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close for trend
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate KAMA(10,2,30) on 12h close
    # Efficiency Ratio (ER) = |change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    # Volatility = sum of absolute changes over ER period
    vol = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    # Avoid division by zero
    er = np.where(vol > 0, change / vol, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume moving average for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_in_trade = 0
    
    for i in range(10, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require volume above 1.5x average
        vol_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_in_trade = 0
            # Time-based exit: max 8 bars
            elif bars_in_trade >= 8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_in_trade = 0
            else:
                signals[i] = 0.25
                bars_in_trade += 1
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_in_trade = 0
            # Time-based exit: max 8 bars
            elif bars_in_trade >= 8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_in_trade = 0
            else:
                signals[i] = -0.25
                bars_in_trade += 1
        else:
            # Look for entries with volume filter
            if vol_filter:
                # Long entry: price crosses above KAMA with 1d uptrend
                if close[i] > kama[i] and close[i-1] <= kama[i-1] and close[i] > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_in_trade = 1
                # Short entry: price crosses below KAMA with 1d downtrend
                elif close[i] < kama[i] and close[i-1] >= kama[i-1] and close[i] < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_in_trade = 1
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA trend filter and volume confirmation.
# Goes long when price breaks above 20-period high with 1w EMA uptrend and volume > average.
# Goes short when price breaks below 20-period low with 1w EMA downtrend and volume > average.
# Uses ATR-based stoploss to limit downside. Designed to work in both bull and bear markets
# by following the trend on higher timeframe (1w) while capturing breakouts on 12h.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.

name = "12h_donchian20_1w_ema_vol_v1"
timeframe = "12h"
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
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w close
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Donchian channels (20-period) on 12h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume moving average for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian low or 1w EMA turns down
            elif close[i] < low_min[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian high or 1w EMA turns up
            elif close[i] > high_max[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume filter
            if vol_filter:
                # Long entry: price breaks above Donchian high with 1w EMA uptrend
                if close[i] > high_max[i] and close[i] > ema_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short entry: price breaks below Donchian low with 1w EMA downtrend
                elif close[i] < low_min[i] and close[i] < ema_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d trend filter and volume confirmation.
# Uses Williams %R(14) on 12h for overbought/oversold signals, 1d EMA50 for trend filter,
# and volume spike (>1.5x average) for confirmation. Goes long when %R crosses above -80
# from oversold with 1d uptrend; short when %R crosses below -20 from overbought with 1d downtrend.
# Includes ATR-based stoploss (2x ATR) and time-based exit (10 bars) to limit losses.
# Designed to work in ranging markets with clear reversal signals.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.

name = "12h_williams1d_vol_v1"
timeframe = "12h"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close for trend
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Williams %R(14) on 12h
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    rr = highest_high - lowest_low
    willr = np.where(rr != 0, (highest_high - close) / rr * -100, -50)
    
    # Volume moving average for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0