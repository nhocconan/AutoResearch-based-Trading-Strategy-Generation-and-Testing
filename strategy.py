#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 4h trend filter and volume confirmation
# Long when price breaks above 12h Donchian upper channel, 4h close > 4h EMA50 (uptrend), and volume > 1.5x 12h average volume
# Short when price breaks below 12h Donchian lower channel, 4h close < 4h EMA50 (downtrend), and volume > 1.5x 12h average volume
# Exit when Donchian breakout reverses or trend changes
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 4h EMA50 for trend filter and 12h volume average for confirmation
# Target: 75-200 total trades over 4 years (19-50/year)

name = "12h_donchian20_4h_ema50_vol_v1"
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
    
    # 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 12h volume average for confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    volume_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
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
            np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_ma_12h_aligned[i]) or 
            np.isnan(atr[i])):
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
            # Exit: Donchian breakout reverses or trend changes
            elif close[i] < donchian_lower[i] or close[i] < ema50_4h_aligned[i]:
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
            # Exit: Donchian breakout reverses or trend changes
            elif close[i] > donchian_upper[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Donchian breakout, trend alignment, and volume confirmation
            # Bullish breakout: price closes above 12h Donchian upper channel
            bullish_breakout = close[i] > donchian_upper[i]
            # Bearish breakout: price closes below 12h Donchian lower channel
            bearish_breakout = close[i] < donchian_lower[i]
            
            # Long: bullish breakout, 4h uptrend, volume spike
            if (bullish_breakout and
                close[i] > ema50_4h_aligned[i] and
                volume[i] > 1.5 * volume_ma_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish breakout, 4h downtrend, volume spike
            elif (bearish_breakout and
                  close[i] < ema50_4h_aligned[i] and
                  volume[i] > 1.5 * volume_ma_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 4h trend filter and volume confirmation
# Long when price breaks above 12h Donchian upper channel, 4h close > 4h EMA50 (uptrend), and volume > 1.5x 12h average volume
# Short when price breaks below 12h Donchian lower channel, 4h close < 4h EMA50 (downtrend), and volume > 1.5x 12h average volume
# Exit when Donchian breakout reverses or trend changes
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 4h EMA50 for trend filter and 12h volume average for confirmation
# Target: 75-200 total trades over 4 years (19-50/year)

name = "12h_donchian20_4h_ema50_vol_v1"
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
    
    # 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 12h volume average for confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    volume_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
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
            np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_ma_12h_aligned[i]) or 
            np.isnan(atr[i])):
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
            # Exit: Donchian breakout reverses or trend changes
            elif close[i] < donchian_lower[i] or close[i] < ema50_4h_aligned[i]:
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
            # Exit: Donchian breakout reverses or trend changes
            elif close[i] > donchian_upper[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Donchian breakout, trend alignment, and volume confirmation
            # Bullish breakout: price closes above 12h Donchian upper channel
            bullish_breakout = close[i] > donchian_upper[i]
            # Bearish breakout: price closes below 12h Donchian lower channel
            bearish_breakout = close[i] < donchian_lower[i]
            
            # Long: bullish breakout, 4h uptrend, volume spike
            if (bullish_breakout and
                close[i] > ema50_4h_aligned[i] and
                volume[i] > 1.5 * volume_ma_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish breakout, 4h downtrend, volume spike
            elif (bearish_breakout and
                  close[i] < ema50_4h_aligned[i] and
                  volume[i] > 1.5 * volume_ma_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals