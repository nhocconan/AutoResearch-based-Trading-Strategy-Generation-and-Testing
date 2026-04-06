#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter, volume confirmation, and ATR stoploss
# Enter long when: price breaks above Donchian(20) high, price > 1d EMA(50), volume > 2.0x 20-period average
# Enter short when: price breaks below Donchian(20) low, price < 1d EMA(50), volume > 2.0x 20-period average
# Exit when opposite Donchian breakout occurs or ATR-based stoploss (2*ATR) is hit
# Target: 15-30 trades/year per symbol (60-120 total over 4 years) with strict entry conditions

name = "4h_donchian20_1d_ema_vol_atr_v1"
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
    
    # Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    # ATR(14) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = high[0] - low[0]  # First TR
    atr = tr.ewm(alpha=1/14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                # Maintain position until stoploss or exit signal
                if position == 1 and close[i] < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR stoploss hit
            if close[i] < donchian_low[i] or close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR stoploss hit
            if close[i] > donchian_high[i] or close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend filter + volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                # Bullish breakout above Donchian high with trend and volume
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif close[i] < donchian_low[i] and close[i] < ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                # Bearish breakout below Donchian low with trend and volume
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter, volume confirmation, and ATR stoploss
# Enter long when: price breaks above Donchian(20) high, price > 1d EMA(50), volume > 2.0x 20-period average
# Enter short when: price breaks below Donchian(20) low, price < 1d EMA(50), volume > 2.0x 20-period average
# Exit when opposite Donchian breakout occurs or ATR-based stoploss (2*ATR) is hit
# Target: 15-30 trades/year per symbol (60-120 total over 4 years) with strict entry conditions

name = "4h_donchian20_1d_ema_vol_atr_v1"
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
    
    # Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    # ATR(14) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = high[0] - low[0]  # First TR
    atr = tr.ewm(alpha=1/14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                # Maintain position until stoploss or exit signal
                if position == 1 and close[i] < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR stoploss hit
            if close[i] < donchian_low[i] or close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR stoploss hit
            if close[i] > donchian_high[i] or close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend filter + volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                # Bullish breakout above Donchian high with trend and volume
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif close[i] < donchian_low[i] and close[i] < ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                # Bearish breakout below Donchian low with trend and volume
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals