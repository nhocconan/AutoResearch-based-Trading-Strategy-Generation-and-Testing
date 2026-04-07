#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour mean reversion with 4-hour trend filter and volume confirmation.
# Long when: 1h RSI < 30 (oversold), 4h close > 4h EMA50 (uptrend), and volume > 1.3x 1h volume average.
# Short when: 1h RSI > 70 (overbought), 4h close < 4h EMA50 (downtrend), and volume > 1.3x 1h volume average.
# Exit when RSI crosses 50 (mean reversion complete) or opposite signal occurs.
# Stoploss at 2.0 * ATR(14).
# Position size: 0.20 (20% of capital).
# Uses 4h EMA50 for trend filter and 1h volume average for confirmation.
# Target: 75-150 total trades over 4 years (19-38/year) to avoid fee drag.

name = "1h_rsi30_4h_ema50_vol_v1"
timeframe = "1h"
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
    
    # 1h RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
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
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI crosses above 50 (mean reversion complete)
            elif rsi[i] > 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI crosses below 50 (mean reversion complete)
            elif rsi[i] < 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: RSI < 30 (oversold), price above EMA50 (uptrend), volume spike
            if (rsi[i] < 30 and
                close[i] > ema_4h_aligned[i] and
                volume[i] > 1.3 * volume_ma[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: RSI > 70 (overbought), price below EMA50 (downtrend), volume spike
            elif (rsi[i] > 70 and
                  close[i] < ema_4h_aligned[i] and
                  volume[i] > 1.3 * volume_ma[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour momentum strategy with 4-hour trend filter and volume confirmation.
# Long when: 1h price > 1h EMA20, 4h close > 4h EMA50 (uptrend), and volume > 1.5x 1h volume average.
# Short when: 1h price < 1h EMA20, 4h close < 4h EMA50 (downtrend), and volume > 1.5x 1h volume average.
# Exit when price crosses EMA20 in opposite direction or trend reverses.
# Stoploss at 2.0 * ATR(14).
# Position size: 0.20 (20% of capital).
# Uses 4h EMA50 for trend filter and 1h EMA20 for entry timing.
# Target: 75-150 total trades over 4 years (19-38/year) to avoid fee drag.

name = "1h_momentum_4h_ema50_vol_v1"
timeframe = "1h"
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
    
    # 1h EMA20 for entry timing
    ema_20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    
    # 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
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
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(ema_20[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below EMA20 or trend reverses
            elif close[i] < ema_20[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above EMA20 or trend reverses
            elif close[i] > ema_20[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price above EMA20, price above EMA50 (uptrend), volume spike
            if (close[i] > ema_20[i] and
                close[i] > ema_4h_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price below EMA20, price below EMA50 (downtrend), volume spike
            elif (close[i] < ema_20[i] and
                  close[i] < ema_4h_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour breakout strategy with 4-hour trend filter and volume confirmation.
# Long when: 1h price breaks above 1h Donchian(10) upper band, 4h close > 4h EMA50 (uptrend), and volume > 1.5x 1h volume average.
# Short when: 1h price breaks below 1h Donchian(10) lower band, 4h close < 4h EMA50 (downtrend), and volume > 1.5x 1h volume average.
# Exit when price crosses Donchian midpoint or trend reverses.
# Stoploss at 2.0 * ATR(14).
# Position size: 0.20 (20% of capital).
# Uses 4h EMA50 for trend filter and 1h Donchian(10) for breakout signals.
# Target: 75-150 total trades over 4 years (19-38/year) to avoid fee drag.

name = "1h_donchian10_4h_ema50_vol_v1"
timeframe = "1h"
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
    
    # 1h Donchian(10) channels
    high_series = pd.Series(high)
    donchian_upper = high_series.rolling(window=10, min_periods=10).max().values
    low_series = pd.Series(low)
    donchian_lower = low_series.rolling(window=10, min_periods=10).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
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
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below midpoint or trend reverses
            elif close[i] < donchian_mid[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above midpoint or trend reverses
            elif close[i] > donchian_mid[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above upper band, price above EMA50 (uptrend), volume spike
            if (close[i] > donchian_upper[i] and
                close[i] > ema_4h_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, price below EMA50 (downtrend), volume spike
            elif (close[i] < donchian_lower[i] and
                  close[i] < ema_4h_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour mean reversion with 4-hour trend filter and volume confirmation.
# Long when: 1h RSI < 30 (oversold), 4h close > 4h EMA50 (uptrend), and volume > 1.3x 1h volume average.
# Short when: 1h RSI > 70 (overbought), 4h close < 4h EMA50 (downtrend), and volume > 1.3x 1h volume average.
# Exit when RSI crosses 50 (mean reversion complete) or opposite signal occurs.
# Stoploss at 2.0 * ATR(14).
# Position size: 0.20 (20% of capital).
# Uses 4h EMA50 for trend filter and 1h volume average for confirmation.
# Target: 75-150 total trades over 4 years (19-38/year) to avoid fee drag.

name = "1h_rsi30_4h_ema50_vol_v1"
timeframe = "1h"
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
    
    # 1h RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
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
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI crosses above 50 (mean reversion complete)
            elif rsi[i] > 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI crosses below 50 (mean reversion complete)
            elif rsi[i] < 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: RSI < 30 (oversold), price above EMA50 (uptrend), volume spike
            if (rsi[i] < 30 and
                close[i] > ema_4h_aligned[i] and
                volume[i] > 1.3 * volume_ma[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: RSI > 70 (overbought), price below EMA50 (downtrend), volume spike
            elif (rsi[i] > 70 and
                  close[i] < ema_4h_aligned[i] and
                  volume[i] > 1.3 * volume_ma[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals

---  End of response ---