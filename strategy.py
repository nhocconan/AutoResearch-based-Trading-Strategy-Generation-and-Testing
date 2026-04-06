#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA trend filter
# Long: price breaks above Donchian(20) upper band + volume > 1.5x 20-period average + price > 12h EMA(20)
# Short: price breaks below Donchian(20) lower band + volume > 1.5x 20-period average + price < 12h EMA(20)
# Exit: opposite Donchian breakout or trailing stop at 2x ATR(14)
# Target: 75-200 trades over 4 years by requiring confluence of breakout, volume, and trend

name = "4h_donchian20_vol_ema12_trend_v1"
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
    
    # Donchian(20) on 4h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # 12h EMA(20) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
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
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(volume_threshold[i]) or np.isnan(ema_12h_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                # Trailing stop: exit if price moves 2*ATR against position
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
            # Exit: opposite breakout or stoploss
            if close[i] < low_roll[i] or close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: opposite breakout or stoploss
            if close[i] > high_roll[i] or close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout + volume + trend
            if volume[i] > volume_threshold[i]:
                if close[i] > high_roll[i] and close[i] > ema_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < low_roll[i] and close[i] < ema_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 1d + volume spike + 12h ADX trend filter
# Long: price > Camarilla H3 (1d) + volume > 2.0x 20-period average + 12h ADX > 25
# Short: price < Camarilla L3 (1d) + volume > 2.0x 20-period average + 12h ADX > 25
# Exit: price crosses Camarilla H4/L4 or opposite pivot level touch
# Target: 75-200 trades over 4 years by requiring confluence of pivot break, volume, and trend

name = "4h_camarilla1d_vol_adx12_v1"
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
    
    # Camarilla levels from 1d
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 12
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 12
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 6
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 6
    camarilla_h2 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 4
    camarilla_l2 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 4
    camarilla_h1 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 3
    camarilla_l1 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 3
    
    # Align Camarilla levels to 4h
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    # 12h ADX(14) for trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX components
    plus_dm = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_12h
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_12h
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_12h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(volume_threshold[i]) or np.isnan(adx_12h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price < L4 or opposite H3 touch
            if close[i] < l4_aligned[i] or (close[i] > h3_aligned[i] and close[i-1] <= h3_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price > H4 or opposite L3 touch
            if close[i] > h4_aligned[i] or (close[i] < l3_aligned[i] and close[i-1] >= l3_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: pivot break + volume + trend
            if volume[i] > volume_threshold[i] and adx_12h_aligned[i] > 25:
                if close[i] > h3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < l3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h CRSI(2,1,2) + 12h Choppiness Index regime filter + Donchian(20) exit
# Long: CRSI < 15 AND 12h Choppiness > 61.8 (ranging market) 
# Short: CRSI > 85 AND 12h Choppiness > 61.8 (ranging market)
# Exit: Donchian(20) opposite breakout
# Target: 75-200 trades over 4 years by requiring oversold/overbought in ranging markets

name = "4h_crsi_chop_donchian_exit_v1"
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
    
    # CRSI components: RSI(2), RSI(1), Percent Rank(2)
    # RSI(2)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_2 = 100 - (100 / (1 + rs))
    rsi_2 = rsi_2.values
    
    # RSI(1) - simplified as price change
    rsi_1 = 100 * (close > np.roll(close, 1)).astype(float)
    rsi_1[0] = 50
    
    # Percent Rank(2) - percentage of days price was higher in last 2 periods
    rank_2 = np.zeros_like(close)
    for i in range(2, n):
        window = close[i-2:i+1]
        rank_2[i] = (np.sum(window < close[i]) / len(window)) * 100
    
    # CRSI = (RSI(2) + RSI(1) + PercentRank(2)) / 3
    crsi = (rsi_2 + rsi_1 + rank_2) / 3.0
    
    # 12h Choppiness Index
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TRUE RANGE for last 14 periods
    sum_tr_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) - Min(low) over last 14 periods
    max_h_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_l_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    range_14 = max_h_14 - min_l_14
    
    # Choppiness = 100 * log10(sum_tr_14 / range_14) / log10(14)
    # Avoid division by zero
    chop_ratio = np.where(range_14 > 0, sum_tr_14 / range_14, 1.0)
    chop_12h = 100 * np.log10(chop_ratio) / np.log10(14)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Donchian(20) for exit
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(crsi[i]) or np.isnan(chop_12h_aligned[i]) or 
            np.isnan(high_roll[i]) or np.isnan(low_roll[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Donchian lower break
            if close[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Donchian upper break
            if close[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: CRSI extreme + choppy market (range)
            if chop_12h_aligned[i] > 61.8:  # ranging market
                if crsi[i] < 15:
                    signals[i] = 0.25
                    position = 1
                elif crsi[i] > 85:
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA(10,2,30) direction + RSI(14) filter + 12h Choppiness regime
# Long: KAMA rising AND RSI(14) > 50 AND 12h Choppiness < 38.2 (trending market)
# Short: KAMA falling AND RSI(14) < 50 AND 12h Choppiness < 38.2 (trending market)
# Exit: opposite KAMA direction or RSI crosses 50
# Target: 75-200 trades over 4 years by trading with trend in trending markets

name = "4h_kama_rsi_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    
    # KAMA(10,2,30)
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = 0  # First 10 values
    
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
    # Proper volatility calculation: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    
    er = np.where(volatility > 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 12h Choppiness Index
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TRUE RANGE for last 14 periods
    sum_tr_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) - Min(low) over last 14 periods
    max_h_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_l_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    range_14 = max_h_14 - min_l_14
    
    # Choppiness = 100 * log10(sum_tr_14 / range_14) / log10(14)
    chop_ratio = np.where(range_14 > 0, sum_tr_14 / range_14, 1.0)
    chop_12h = 100 * np.log10(chop_ratio) / np.log10(14)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_12h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: KAMA falling or RSI < 50
            if kama[i] < kama[i-1] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: KAMA rising or RSI > 50
            if kama[i] > kama[i-1] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: KAMA direction + RSI filter + chop regime
            if chop_12h_aligned[i] < 38.2:  # trending market
                if kama[i] > kama[i-1] and rsi[i] > 50:
                    signals[i] = 0.25
                    position = 1
                elif kama[i] < kama[i-1] and rsi[i] < 50:
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA trend filter
# Long: price breaks above Donchian(20) upper band + volume > 1.5x 20-period average + price > 12h EMA(20)
# Short: price breaks below Donchian(20) lower band + volume > 1.5x 20-period average + price < 12h EMA(20)
# Exit: opposite Donchian breakout or trailing stop at 2x ATR(14)
# Target: 75-200 trades over 4 years by requiring confluence of breakout, volume, and trend

name = "4h_donchian20_vol_ema12_trend_v1"
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
    
    # Donchian(20) on 4h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # 12h EMA(20) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
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
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(volume_threshold[i]) or np.isnan(ema_12h_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                # Trailing stop: exit if price moves 2*ATR against position
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
            # Exit: opposite breakout or stoploss
            if close[i] < low_roll[i] or close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: opposite breakout or stoploss
            if close[i] > high_roll[i] or close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout + volume + trend
            if volume[i] > volume_threshold[i]:
                if close[i] > high_roll[i] and close[i] > ema_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < low_roll[i] and close[i] < ema_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ADX(14) trend strength + 12h EMA(50) direction + volume confirmation
# Long: ADX > 25 AND 12h EMA(50) rising AND volume > 1.5x 20-period average
# Short: ADX > 25 AND 12h EMA(50) falling AND volume > 1.5x 20-period average
# Exit: ADX < 20 (trend weakening) or opposite EMA direction
# Target: 75-200 trades over 4 years by trading strong trends with volume confirmation

name = "4h_adx14_ema50_12h_vol_v1"
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
    
    # ADX(14) calculation
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series