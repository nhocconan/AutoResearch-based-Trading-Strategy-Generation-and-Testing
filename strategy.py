#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(40) breakout with 1-day EMA(200) trend filter and volume confirmation
# Long when price breaks above Donchian(40) high AND price > 1d EMA(200) AND volume > 2x 20-period avg
# Short when price breaks below Donchian(40) low AND price < 1d EMA(200) AND volume > 2x 20-period avg
# Exit when price crosses below/above 1d EMA(200) or opposite Donchian break
# Uses higher threshold (Donchian 40) and volume multiplier (2x) to reduce trades
# EMA(200) provides strong trend filter for both bull and bear markets
# Target: 50-150 trades over 4 years (12-38/year) to minimize fee drag

name = "4h_donchian40_1d_ema200_vol_v1"
timeframe = "4h"
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
    
    # 1d data for EMA(200) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(200) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h Donchian channels (40-period) - higher threshold for fewer trades
    donchian_high = pd.Series(high).rolling(window=40, min_periods=40).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=40, min_periods=40).min().shift(1).values
    
    # Volume average (20-period)
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
    
    for i in range(40, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below EMA(200) or breaks below Donchian low
            elif close[i] < ema_1d_aligned[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above EMA(200) or breaks above Donchian high
            elif close[i] > ema_1d_aligned[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above Donchian high, price above EMA(200) (bullish trend), volume spike
            if (close[i] > donchian_high[i] and
                close[i] > ema_1d_aligned[i] and
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low, price below EMA(200) (bearish trend), volume spike
            elif (close[i] < donchian_low[i] and
                  close[i] < ema_1d_aligned[i] and
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(40) breakout with 1-day EMA(200) trend filter and volume confirmation
# Long when price breaks above Donchian(40) high AND price > 1d EMA(200) AND volume > 2x 20-period avg
# Short when price breaks below Donchian(40) low AND price < 1d EMA(200) AND volume > 2x 20-period avg
# Exit when price crosses below/above 1d EMA(200) or opposite Donchian break
# Uses higher threshold (Donchian 40) and volume multiplier (2x) to reduce trades
# EMA(200) provides strong trend filter for both bull and bear markets
# Target: 50-150 trades over 4 years (12-38/year) to minimize fee drag

name = "4h_donchian40_1d_ema200_vol_v1"
timeframe = "4h"
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
    
    # 1d data for EMA(200) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(200) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h Donchian channels (40-period) - higher threshold for fewer trades
    donchian_high = pd.Series(high).rolling(window=40, min_periods=40).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=40, min_periods=40).min().shift(1).values
    
    # Volume average (20-period)
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
    
    for i in range(40, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below EMA(200) or breaks below Donchian low
            elif close[i] < ema_1d_aligned[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above EMA(200) or breaks above Donchian high
            elif close[i] > ema_1d_aligned[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above Donchian high, price above EMA(200) (bullish trend), volume spike
            if (close[i] > donchian_high[i] and
                close[i] > ema_1d_aligned[i] and
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low, price below EMA(200) (bearish trend), volume spike
            elif (close[i] < donchian_low[i] and
                  close[i] < ema_1d_aligned[i] and
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals