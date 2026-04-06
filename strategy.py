#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h/1d trend filter and volume confirmation
# Long when 4h EMA21 > 1d EMA50 (bullish trend) AND 1h RSI(14) > 50 AND volume > 1.5x average
# Short when 4h EMA21 < 1d EMA50 (bearish trend) AND 1h RSI(14) < 50 AND volume > 1.5x average
# Exit when trend reverses or RSI crosses back to neutral
# Uses 4h/1d for trend direction, 1h for entry timing
# Target: 60-150 trades over 4 years (15-37/year) to avoid fee drag
# Works in bull markets (follows trend) and bear markets (shorts trend)

name = "1h_momentum_4h1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (inclusive)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h EMA21 for trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA50 for trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend condition: 4h EMA21 vs 1d EMA50
        bullish_trend = ema_4h_aligned[i] > ema_1d_aligned[i]
        bearish_trend = ema_4h_aligned[i] < ema_1d_aligned[i]
        
        # Exit conditions
        if position == 1:  # long position
            if not bullish_trend or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if not bearish_trend or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries
            # Long: bullish trend + RSI > 50 (bullish momentum) + volume confirmation
            if bullish_trend and rsi[i] > 50 and volume[i] > volume_threshold[i]:
                signals[i] = 0.20
                position = 1
            # Short: bearish trend + RSI < 50 (bearish momentum) + volume confirmation
            elif bearish_trend and rsi[i] < 50 and volume[i] > volume_threshold[i]:
                signals[i] = -0.20
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h/1d trend filter and volume confirmation
# Long when 4h EMA21 > 1d EMA50 (bullish trend) AND 1h RSI(14) > 50 AND volume > 1.5x average
# Short when 4h EMA21 < 1d EMA50 (bearish trend) AND 1h RSI(14) < 50 AND volume > 1.5x average
# Exit when trend reverses or RSI crosses back to neutral
# Uses 4h/1d for trend direction, 1h for entry timing
# Target: 60-150 trades over 4 years (15-37/year) to avoid fee drag
# Works in bull markets (follows trend) and bear markets (shorts trend)

name = "1h_momentum_4h1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (inclusive)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h EMA21 for trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA50 for trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend condition: 4h EMA21 vs 1d EMA50
        bullish_trend = ema_4h_aligned[i] > ema_1d_aligned[i]
        bearish_trend = ema_4h_aligned[i] < ema_1d_aligned[i]
        
        # Exit conditions
        if position == 1:  # long position
            if not bullish_trend or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if not bearish_trend or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries
            # Long: bullish trend + RSI > 50 (bullish momentum) + volume confirmation
            if bullish_trend and rsi[i] > 50 and volume[i] > volume_threshold[i]:
                signals[i] = 0.20
                position = 1
            # Short: bearish trend + RSI < 50 (bearish momentum) + volume confirmation
            elif bearish_trend and rsi[i] < 50 and volume[i] > volume_threshold[i]:
                signals[i] = -0.20
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h/1d trend filter and volume confirmation
# Long when 4h EMA21 > 1d EMA50 (bullish trend) AND 1h RSI(14) > 50 AND volume > 1.5x average
# Short when 4h EMA21 < 1d EMA50 (bearish trend) AND 1h RSI(14) < 50 AND volume > 1.5x average
# Exit when trend reverses or RSI crosses back to neutral
# Uses 4h/1d for trend direction, 1h for entry timing
# Target: 60-150 trades over 4 years (15-37/year) to avoid fee drag
# Works in bull markets (follows trend) and bear markets (shorts trend)

name = "1h_momentum_4h1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (inclusive)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h EMA21 for trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA50 for trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend condition: 4h EMA21 vs 1d EMA50
        bullish_trend = ema_4h_aligned[i] > ema_1d_aligned[i]
        bearish_trend = ema_4h_aligned[i] < ema_1d_aligned[i]
        
        # Exit conditions
        if position == 1:  # long position
            if not bullish_trend or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if not bearish_trend or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries
            # Long: bullish trend + RSI > 50 (bullish momentum) + volume confirmation
            if bullish_trend and rsi[i] > 50 and volume[i] > volume_threshold[i]:
                signals[i] = 0.20
                position = 1
            # Short: bearish trend + RSI < 50 (bearish momentum) + volume confirmation
            elif bearish_trend and rsi[i] < 50 and volume[i] > volume_threshold[i]:
                signals[i] = -0.20
                position = -1
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h/1d trend filter and volume confirmation
# Long when 4h EMA21 > 1d EMA50 (bullish trend) AND 1h RSI(14) > 50 AND volume > 1.5x average
# Short when 4h EMA21 < 1d EMA50 (bearish trend) AND 1h RSI(14) < 50 AND volume > 1.5x average
# Exit when trend reverses or RSI crosses back to neutral
# Uses 4h/1d for trend direction, 1h for entry timing
# Target: 60-150 trades over 4 years (15-37/year) to avoid fee drag
# Works in bull markets (follows trend) and bear markets (shorts trend)

name = "1h_momentum_4h1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (inclusive)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h EMA21 for trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA50 for trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend condition: 4h EMA21 vs 1d EMA50
        bullish_trend = ema_4h_aligned[i] > ema_1d_aligned[i]
        bearish_trend = ema_4h_aligned[i] < ema_1d_aligned[i]
        
        # Exit conditions
        if position == 1:  # long position
            if not bullish_trend or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if not bearish_trend or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries
            # Long: bullish trend + RSI > 50 (bullish momentum) + volume confirmation
            if bullish_trend and rsi[i] > 50 and volume[i] > volume_threshold[i]:
                signals[i] = 0.20
                position = 1
            # Short: bearish trend + RSI < 50 (bearish momentum) + volume confirmation
            elif bearish_trend and rsi[i] < 50 and volume[i] > volume_threshold[i]:
                signals[i] = -0.20
                position = -1
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h/1d trend filter and volume confirmation
# Long when 4h EMA21 > 1d EMA50 (bullish trend) AND 1h RSI(14) > 50 AND volume > 1.5x average
# Short when 4h EMA21 < 1d EMA50 (bearish trend) AND 1h RSI(14) < 50 AND volume > 1.5x average
# Exit when trend reverses or RSI crosses back to neutral
# Uses 4h/1d for trend direction, 1h for entry timing
# Target: 60-150 trades over 4 years (15-37/year) to avoid fee drag
# Works in bull markets (follows trend) and bear markets (shorts trend)

name = "1h_momentum_4h1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (inclusive)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h EMA21 for trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA50 for trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend condition: 4h EMA21 vs 1d EMA50
        bullish_trend = ema_4h_aligned[i] > ema_1d_aligned[i]
        bearish_trend = ema_4h_aligned[i] < ema_1d_aligned[i]
        
        # Exit conditions
        if position == 1:  # long position
            if not bullish_trend or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if not bearish_trend or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries
            # Long: bullish trend + RSI > 50 (bullish momentum) + volume confirmation
            if bullish_trend and rsi[i] > 50 and volume[i] > volume_threshold[i]:
                signals[i] = 0.20
                position = 1
            # Short: bearish trend + RSI < 50 (bearish momentum) + volume confirmation
            elif bearish_trend and rsi[i] < 50 and volume[i] > volume_threshold[i]:
                signals[i] = -0.20
                position = -1
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h/1d trend filter and volume confirmation
# Long when 4h EMA21 > 1d EMA50 (bullish trend) AND 1h RSI(14) > 50 AND volume > 1.5x average
# Short when 4h EMA21 < 1d EMA50 (bearish trend) AND 1h RSI(14) < 50 AND volume > 1.5x average
# Exit when trend reverses or RSI crosses back to neutral
# Uses 4h/1d for trend direction, 1h for entry timing
# Target: 60-150 trades over 4 years (15-37/year) to avoid fee drag
# Works in bull markets (follows trend) and bear markets (shorts trend)

name = "1h_momentum_4h1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (inclusive)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h EMA21 for trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA50 for trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend condition: 4h EMA21 vs 1d EMA50
        bullish_trend = ema_4h_aligned[i] > ema_1d_aligned[i]
        bearish_trend = ema_4h_aligned[i] < ema_1d_aligned[i]
        
        # Exit conditions
        if position == 1:  # long position
            if not bullish_trend or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if not bearish_trend or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries
            # Long: bullish trend + RSI > 50 (bullish momentum) + volume confirmation
            if bullish_trend and rsi[i] > 50 and volume[i] > volume_threshold[i]:
                signals[i] = 0.20
                position = 1
            # Short: bearish trend + RSI < 50 (bearish momentum) + volume confirmation
            elif bearish_trend and rsi[i] < 5