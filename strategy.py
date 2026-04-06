#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d EMA trend + volume confirmation
# Long when price breaks above Donchian(20) high AND price > 1d EMA(50) AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND price < 1d EMA(50) AND volume > 1.5x average
# Exit when price crosses back to 1d EMA(50) or trailing stop at 2x ATR
# Targets 75-200 total trades over 4 years (19-50/year) by requiring confluence of trend, breakout, and volume

name = "4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean()
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    # ATR (14-period) for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        # Update trailing stop for existing positions
        if position == 1:  # long position
            # Exit if price crosses below EMA(50) or hits trailing stop
            if close[i] < ema_50_aligned[i] or close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Exit if price crosses above EMA(50) or hits trailing stop
            if close[i] > ema_50_aligned[i] or close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
        else:
            # Look for entries: Donchian breakout + EMA trend + volume confirmation
            # Long: price breaks above Donchian high AND price > EMA(50) AND volume confirmation
            if (close[i] > donch_high[i] and close[i] > ema_50_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low AND price < EMA(50) AND volume confirmation
            elif (close[i] < donch_low[i] and close[i] < ema_50_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d EMA trend + volume confirmation
# Long when price breaks above Donchian(20) high AND price > 1d EMA(50) AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND price < 1d EMA(50) AND volume > 1.5x average
# Exit when price crosses back to 1d EMA(50) or trailing stop at 2x ATR
# Targets 75-200 total trades over 4 years (19-50/year) by requiring confluence of trend, breakout, and volume

name = "4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean()
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    # ATR (14-period) for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        # Update trailing stop for existing positions
        if position == 1:  # long position
            # Exit if price crosses below EMA(50) or hits trailing stop
            if close[i] < ema_50_aligned[i] or close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Exit if price crosses above EMA(50) or hits trailing stop
            if close[i] > ema_50_aligned[i] or close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
        else:
            # Look for entries: Donchian breakout + EMA trend + volume confirmation
            # Long: price breaks above Donchian high AND price > EMA(50) AND volume confirmation
            if (close[i] > donch_high[i] and close[i] > ema_50_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low AND price < EMA(50) AND volume confirmation
            elif (close[i] < donch_low[i] and close[i] < ema_50_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d EMA trend + volume confirmation
# Long when price breaks above Donchian(20) high AND price > 1d EMA(50) AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND price < 1d EMA(50) AND volume > 1.5x average
# Exit when price crosses back to 1d EMA(50) or trailing stop at 2x ATR
# Targets 75-200 total trades over 4 years (19-50/year) by requiring confluence of trend, breakout, and volume

name = "4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean()
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    # ATR (14-period) for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        # Update trailing stop for existing positions
        if position == 1:  # long position
            # Exit if price crosses below EMA(50) or hits trailing stop
            if close[i] < ema_50_aligned[i] or close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Exit if price crosses above EMA(50) or hits trailing stop
            if close[i] > ema_50_aligned[i] or close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
        else:
            # Look for entries: Donchian breakout + EMA trend + volume confirmation
            # Long: price breaks above Donchian high AND price > EMA(50) AND volume confirmation
            if (close[i] > donch_high[i] and close[i] > ema_50_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low AND price < EMA(50) AND volume confirmation
            elif (close[i] < donch_low[i] and close[i] < ema_50_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d EMA trend + volume confirmation
# Long when price breaks above Donchian(20) high AND price > 1d EMA(50) AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND price < 1d EMA(50) AND volume > 1.5x average
# Exit when price crosses back to 1d EMA(50) or trailing stop at 2x ATR
# Targets 75-200 total trades over 4 years (19-50/year) by requiring confluence of trend, breakout, and volume

name = "4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean()
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    # ATR (14-period) for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        # Update trailing stop for existing positions
        if position == 1:  # long position
            # Exit if price crosses below EMA(50) or hits trailing stop
            if close[i] < ema_50_aligned[i] or close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Exit if price crosses above EMA(50) or hits trailing stop
            if close[i] > ema_50_aligned[i] or close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
        else:
            # Look for entries: Donchian breakout + EMA trend + volume confirmation
            # Long: price breaks above Donchian high AND price > EMA(50) AND volume confirmation
            if (close[i] > donch_high[i] and close[i] > ema_50_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low AND price < EMA(50) AND volume confirmation
            elif (close[i] < donch_low[i] and close[i] < ema_50_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d EMA trend + volume confirmation
# Long when price breaks above Donchian(20) high AND price > 1d EMA(50) AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND price < 1d EMA(50) AND volume > 1.5x average
# Exit when price crosses back to 1d EMA(50) or trailing stop at 2x ATR
# Targets 75-200 total trades over 4 years (19-50/year) by requiring confluence of trend, breakout, and volume

name = "4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean()
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    # ATR (14-period) for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        # Update trailing stop for existing positions
        if position == 1:  # long position
            # Exit if price crosses below EMA(50) or hits trailing stop
            if close[i] < ema_50_aligned[i] or close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Exit if price crosses above EMA(50) or hits trailing stop
            if close[i] > ema_50_aligned[i] or close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
        else:
            # Look for entries: Donchian breakout + EMA trend + volume confirmation
            # Long: price breaks above Donchian high AND price > EMA(50) AND volume confirmation
            if (close[i] > donch_high[i] and close[i] > ema_50_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low AND price < EMA(50) AND volume confirmation
            elif (close[i] < donch_low[i] and close[i] < ema_50_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d EMA trend + volume confirmation
# Long when price breaks above Donchian(20) high AND price > 1d EMA(50) AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND price < 1d EMA(50) AND volume > 1.5x average
# Exit when price crosses back to 1d EMA(50) or trailing stop at 2x ATR
# Targets 75-200 total trades over 4 years (19-50/year) by requiring confluence of trend, breakout, and volume

name = "4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean()
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    # ATR (14-period) for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        # Update trailing stop for existing positions
        if position == 1:  # long position
            # Exit if price crosses below EMA(50) or hits trailing stop
            if close[i] < ema_50_aligned[i] or close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Exit if price crosses above EMA(50) or hits trailing stop
            if close[i] > ema_50_aligned[i] or close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
        else:
            # Look for entries: Donchian breakout + EMA trend + volume confirmation
            # Long: price breaks above Donchian high AND price > EMA(50) AND volume confirmation
            if (close[i] > donch_high[i] and close[i] > ema_50_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low AND price < EMA(50) AND volume confirmation
            elif (close[i] < donch_low[i] and close[i] < ema_50_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d EMA trend + volume confirmation
# Long when price breaks above Donchian(20) high AND price > 1d EMA(50) AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND price < 1d EMA(50) AND volume > 1.5x average
# Exit when price crosses back to 1d EMA(50) or trailing stop at 2x ATR
# Targets 75-200 total trades over 4 years (19-50/year) by requiring confluence of trend, breakout, and volume

name = "4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean()
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    # ATR (14-period) for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        # Update trailing stop for existing positions
        if position == 1:  # long position
            # Exit if price crosses below EMA(50) or hits trailing stop
            if close[i] < ema_50_aligned[i] or close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position