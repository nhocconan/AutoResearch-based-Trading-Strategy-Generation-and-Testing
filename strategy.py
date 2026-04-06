#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 12h EMA trend filter and volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and increasing AND Bear Power < 0 AND price > 12h EMA50
# Short when Bear Power < 0 and decreasing AND Bull Power > 0 AND price < 12h EMA50
# Uses 6h timeframe for balance of signal frequency and noise reduction
# 12h EMA provides higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation reduces false signals
# Target: 75-200 total trades over 4 years (19-50/year) for optimal 6h performance

name = "6h_elder_ray_12h_ema_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray Index components (13-period EMA)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 12h EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: Elder Ray divergence or price crosses 12h EMA
        if position == 1:  # long position
            if bull_power[i] <= 0 or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if bear_power[i] >= 0 or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: Bull Power > 0 AND increasing AND Bear Power < 0 AND price > 12h EMA AND volume confirmation
            if (bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and 
                bear_power[i] < 0 and close[i] > ema50_12h_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND decreasing AND Bull Power > 0 AND price < 12h EMA AND volume confirmation
            elif (bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and 
                  bull_power[i] > 0 and close[i] < ema50_12h_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 1d ATR filter and volume confirmation
# Long when price breaks above Donchian upper (20-period) AND ATR(14) > 1.5x MA(50) AND volume > 2x 20-period average
# Short when price breaks below Donchian lower (20-period) AND ATR(14) > 1.5x MA(50) AND volume > 2x 20-period average
# Exit when price crosses Donchian midline or ATR drops below threshold
# Uses 6h timeframe to balance signal frequency and noise
# ATR filter ensures we only trade in volatile conditions, avoiding choppy markets
# Volume confirmation adds confirmation to breakouts
# Target: 75-200 total trades over 4 years (19-50/year) for optimal 6h performance

name = "6h_donchian20_1d_atr_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # 1-day ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d)
    tr2 = pd.Series(close_1d).shift(1) - pd.Series(high_1d)
    tr3 = pd.Series(close_1d).shift(1) - pd.Series(low_1d)
    tr1_abs = abs(tr1)
    tr2_abs = abs(tr2)
    tr3_abs = abs(tr3)
    true_range = pd.concat([tr1_abs, tr2_abs, tr3_abs], axis=1).max(axis=1)
    atr_1d = true_range.ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # ATR threshold: 1.5x 50-period MA of ATR
    atr_ma = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean()
    atr_threshold = 1.5 * atr_ma.values
    atr_threshold_aligned = align_htf_to_ltf(prices, df_1d, atr_threshold)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 2.0 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(atr_threshold_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price crosses Donchian midline OR volatility drops
        if position == 1:  # long position
            if close[i] < donchian_mid[i] or atr_1d[i] < atr_threshold_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > donchian_mid[i] or atr_1d[i] < atr_threshold_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volatility filter and volume confirmation
            # Long: price breaks above Donchian upper AND ATR above threshold AND volume confirmation
            if (close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1] and 
                atr_1d[i] > atr_threshold_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND ATR above threshold AND volume confirmation
            elif (close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1] and 
                  atr_1d[i] > atr_threshold_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with 1d trend filter and volume confirmation
# Camarilla levels calculated from previous 1d bar: H4 = Close + 1.5*(High-Low), L4 = Close - 1.5*(High-Low)
# Long when price crosses above L4 with closing price > L4 AND price > 1d EMA50 AND volume > 1.5x average
# Short when price crosses below H4 with closing price < H4 AND price < 1d EMA50 AND volume > 1.5x average
# Exit when price reaches opposite H3/L3 level or crosses 1d EMA50
# Uses 6h timeframe for optimal signal frequency
# Camarilla levels provide precise intraday support/resistance
# 1d EMA50 filters for higher timeframe trend
# Volume confirmation reduces false breakouts
# Target: 75-200 total trades over 4 years (19-50/year) for optimal 6h performance

name = "6h_camarilla_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for Camarilla calculation and EMA filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # H4 = Close + 1.5*(High-Low), L4 = Close - 1.5*(High-Low)
    # H3 = Close + 1.125*(High-Low), L3 = Close - 1.125*(High-Low)
    # H2 = Close + 0.75*(High-Low), L2 = Close - 0.75*(High-Low)
    # H1 = Close + 0.5*(High-Low), L1 = Close - 0.5*(High-Low)
    range_1d = high_1d - low_1d
    h4 = close_1d + 1.5 * range_1d
    l4 = close_1d - 1.5 * range_1d
    h3 = close_1d + 1.125 * range_1d
    l3 = close_1d - 1.125 * range_1d
    
    # Align Camarilla levels to 6h timeframe (using previous day's levels)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # 1-day EMA(50) trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price reaches opposite level or crosses 1d EMA
        if position == 1:  # long position
            if close[i] < l3_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > h3_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: price crosses above L4 with close > L4 AND price > 1d EMA50 AND volume confirmation
            if (close[i] > l4_aligned[i] and close[i-1] <= l4_aligned[i-1] and 
                close[i] > ema50_1d_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below H4 with close < H4 AND price < 1d EMA50 AND volume confirmation
            elif (close[i] < h4_aligned[i] and close[i-1] >= h4_aligned[i-1] and 
                  close[i] < ema50_1d_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Tenkan-sen (9-period), Kijun-sen (26-period), Senkou Span A/B (26-period displaced)
# Long when Tenkan > Kijun AND price > Cloud AND price > 1d EMA50 AND volume > 2x average
# Short when Tenkan < Kijun AND price < Cloud AND price < 1d EMA50 AND volume > 2x average
# Exit when Tenkan/Kijun cross reverses or price enters cloud
# Uses 6h timeframe for balance of signal and noise
# Ichimoku provides comprehensive support/resistance and momentum
# 1d EMA50 filters for higher timeframe trend
# Volume confirmation adds validity to signals
# Target: 75-200 total trades over 4 years (19-50/year) for optimal 6h performance

name = "6h_ichimoku_1d_ema_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 displaced 26 periods
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 displaced 26 periods
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # 1-day EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 2.0 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after Ichimoku warmup
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_a[i], senkou_b[i])
        lower_cloud = np.minimum(senkou_a[i], senkou_b[i])
        
        # Check exits: Tenkan/Kijun cross reverse or price enters cloud
        if position == 1:  # long position
            if tenkan[i] < kijun[i] or (close[i] < upper_cloud and close[i] > lower_cloud):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if tenkan[i] > kijun[i] or (close[i] < upper_cloud and close[i] > lower_cloud):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: Tenkan > Kijun AND price > Cloud AND price > 1d EMA50 AND volume confirmation
            if (tenkan[i] > kijun[i] and close[i] > upper_cloud and 
                close[i] > ema50_1d_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan < Kijun AND price < Cloud AND price < 1d EMA50 AND volume confirmation
            elif (tenkan[i] < kijun[i] and close[i] < lower_cloud and 
                  close[i] < ema50_1d_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d ADX trend filter and volume confirmation
# Williams %R: (Highest High - Close)/(Highest High - Lowest Low) * -100
# Long when Williams %R crosses above -80 from below AND ADX(14) > 25 AND volume > 1.5x average
# Short when Williams %R crosses below -20 from above AND ADX(14) > 25 AND volume > 1.5x average
# Exit when Williams %R reaches opposite extreme (-20 for long, -80 for short) or ADX < 20
# Uses 6h timeframe for optimal signal frequency
# Williams %R identifies overbought/oversold conditions
# 1d ADX filters for trending markets only (avoids chop)
# Volume confirmation reduces false signals
# Target: 75-200 total trades over 4 years (19-50/year) for optimal 6h performance

name = "6h_williamsr_1d_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # 1-day ADX(14) trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and Directional Movement
    plus_dm = pd.Series(high_1d).diff()
    minus_dm = pd.Series(low_1d).diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d)
    tr2 = pd.Series(close_1d).shift(1) - pd.Series(high_1d)
    tr3 = pd.Series(close_1d).shift(1) - pd.Series(low_1d)
    tr1_abs = abs(tr1)
    tr2_abs = abs(tr2)
    tr3_abs = abs(tr3)
    true_range = pd.concat([tr1_abs, tr2_abs, tr3_abs], axis=1).max(axis=1)
    
    # Smooth the values
    atr_1d = true_range.ewm(span=14, min_periods=14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=14, min_periods=14, adjust=False).mean() / atr_1d)
    minus_di = 100 * (abs(minus_dm.ewm(span=14, min_periods=14, adjust=False).mean()) / atr_1d)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_1d = dx.ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Align ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(williams_r[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: Williams %R reaches opposite extreme or ADX < 20
        if position == 1:  # long position
            if williams_r[i] > -20 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if williams_r[i] < -80 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: Williams %R crosses above -80 from below AND ADX > 25 AND volume confirmation
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                adx_1d_aligned[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND ADX > 25 AND volume confirmation
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  adx_1d_aligned[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 12h EMA trend filter and volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and increasing AND Bear Power < 0 AND price > 12h EMA50
# Short when Bear Power < 0 and decreasing AND Bull Power > 0 AND price < 12h EMA50
# Uses 6h timeframe for balance of signal frequency and noise reduction
# 12h EMA provides higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation reduces false signals
# Target: 75-200 total trades over 4 years (19-50/year) for optimal 6h performance

name = "6h_elder_ray_12h_ema_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray Index components (13-period EMA)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 12h EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: Elder Ray divergence or price crosses 12h EMA
        if position == 1:  # long position
            if bull_power[i] <= 0 or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if bear_power[i] >= 0 or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: Bull Power > 0 AND increasing AND Bear Power < 0 AND price > 12h EMA AND volume confirmation
            if (bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and 
                bear_power[i] < 0 and close[i] > ema50_12h_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND decreasing AND Bull Power > 0 AND price < 12h EMA AND volume confirmation
            elif (bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and 
                  bull_power[i] > 0 and close[i] < ema50_12h_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 1d ATR filter and volume confirmation
# Long