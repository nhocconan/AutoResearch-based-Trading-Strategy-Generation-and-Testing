#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour KAMA trend with 1-day RSI filter and volume confirmation
# Long when KAMA trending up, 1d RSI < 40 (momentum pullback), and volume > 1.5x 12h average volume
# Short when KAMA trending down, 1d RSI > 60 (overbought rally), and volume > 1.5x 12h average volume
# Exit when KAMA trend reverses or RSI reaches opposite extreme
# Stoploss at 2.5 * ATR(20)
# Position size: 0.25 (25% of capital)
# Uses 1-day RSI for momentum filter and 12h KAMA for trend
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_kama_1d_rsi_vol_v1"
timeframe = "12h"
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
    
    # 12h KAMA for trend
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close - np.roll(close, 10))
    change[0] = 0
    for i in range(1, 10):
        change[i] = abs(close[i] - close[i-10]) if i >= 10 else abs(close[i] - close[0])
    volatility = abs(close - np.roll(close, 1))
    volatility[0] = 0
    er = change / (pd.Series(volatility).rolling(window=10, min_periods=1).sum().values + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1-day data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 12h volume average for confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(20) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
            # Exit: KAMA trend reverses or RSI overbought
            elif close[i] < kama[i] or rsi_1d_aligned[i] > 70:
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
            # Exit: KAMA trend reverses or RSI oversold
            elif close[i] > kama[i] or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with KAMA trend, RSI extreme, and volume confirmation
            # Long: KAMA trending up (price > KAMA), RSI oversold, volume spike
            if (close[i] > kama[i] and
                rsi_1d_aligned[i] < 40 and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: KAMA trending down (price < KAMA), RSI overbought, volume spike
            elif (close[i] < kama[i] and
                  rsi_1d_aligned[i] > 60 and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour KAMA trend with 1-day RSI filter and volume confirmation
# Long when KAMA trending up, 1d RSI < 40 (momentum pullback), and volume > 1.5x 12h average volume
# Short when KAMA trending down, 1d RSI > 60 (overbought rally), and volume > 1.5x 12h average volume
# Exit when KAMA trend reverses or RSI reaches opposite extreme
# Stoploss at 2.5 * ATR(20)
# Position size: 0.25 (25% of capital)
# Uses 1-day RSI for momentum filter and 12h KAMA for trend
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_kama_1d_rsi_vol_v1"
timeframe = "12h"
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
    
    # 12h KAMA for trend
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close - np.roll(close, 10))
    change[0] = 0
    for i in range(1, 10):
        change[i] = abs(close[i] - close[i-10]) if i >= 10 else abs(close[i] - close[0])
    volatility = abs(close - np.roll(close, 1))
    volatility[0] = 0
    er = change / (pd.Series(volatility).rolling(window=10, min_periods=1).sum().values + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1-day data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 12h volume average for confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(20) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
            # Exit: KAMA trend reverses or RSI overbought
            elif close[i] < kama[i] or rsi_1d_aligned[i] > 70:
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
            # Exit: KAMA trend reverses or RSI oversold
            elif close[i] > kama[i] or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with KAMA trend, RSI extreme, and volume confirmation
            # Long: KAMA trending up (price > KAMA), RSI oversold, volume spike
            if (close[i] > kama[i] and
                rsi_1d_aligned[i] < 40 and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: KAMA trending down (price < KAMA), RSI overbought, volume spike
            elif (close[i] < kama[i] and
                  rsi_1d_aligned[i] > 60 and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals