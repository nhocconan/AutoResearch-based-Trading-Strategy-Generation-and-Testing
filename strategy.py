#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA filter and volume confirmation
# Long when Williams %R < -80 (oversold) + price > 1d EMA + volume > 1.5x avg
# Short when Williams %R > -20 (overbought) + price < 1d EMA + volume > 1.5x avg
# Exit when Williams %R crosses -50 or price crosses 1d EMA
# Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Williams %R is effective in ranging markets; EMA filter adds trend bias to avoid whipsaws

name = "6h_williamsr_1d_ema_vol_v1"
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
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # 1d EMA (21-period)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Williams %R crosses -50 or price crosses 1d EMA
        if position == 1:  # long position
            if williams_r[i] >= -50 or close[i] <= ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if williams_r[i] <= -50 or close[i] >= ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for mean reversion with EMA filter and volume
            # Long: oversold + above EMA + volume
            if (williams_r[i] < -80 and 
                close[i] > ema_1d_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: overbought + below EMA + volume
            elif (williams_r[i] > -20 and 
                  close[i] < ema_1d_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA filter and volume confirmation
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Long when Bull Power > 0 and increasing + price > 1d EMA(50) + volume > 1.5x avg
# Short when Bear Power > 0 and increasing + price < 1d EMA(50) + volume > 1.5x avg
# Exit when power turns negative or price crosses 1d EMA
# Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Elder Ray measures bull/bear strength; EMA filter adds trend context to avoid counter-trend trades

name = "6h_elder_ray_1d_ema_vol_v1"
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
    
    # EMA(13) for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(ema13[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: power turns negative or price crosses 1d EMA
        if position == 1:  # long position
            if bull_power[i] <= 0 or close[i] <= ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if bear_power[i] <= 0 or close[i] >= ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for trending moves with EMA filter and volume
            # Long: Bull Power positive and rising + above EMA + volume
            if (bull_power[i] > 0 and 
                bull_power[i] > bull_power[i-1] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power positive and rising + below EMA + volume
            elif (bear_power[i] > 0 and 
                  bear_power[i] > bear_power[i-1] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h ADX filter and volume confirmation
# Long when price breaks above Donchian high + ADX > 25 + volume > 1.5x avg
# Short when price breaks below Donchian low + ADX > 25 + volume > 1.5x avg
# Exit when price crosses Donchian midpoint or ADX < 20
# Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Donchian captures breakouts; ADX filters for trending markets to avoid range whipsaws

name = "6h_donchian_12h_adx_vol_v1"
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
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # ADX (14-period) from 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = np.zeros_like(high_12h)
    minus_dm = np.zeros_like(high_12h)
    plus_dm[0] = 0
    minus_dm[0] = 0
    for i in range(1, len(high_12h)):
        up = high_12h[i] - high_12h[i-1]
        down = low_12h[i-1] - low_12h[i]
        if up > down and up > 0:
            plus_dm[i] = up
        elif down > up and down > 0:
            minus_dm[i] = down
    
    # Smoothed values
    def smooth(val, period):
        smoothed = np.zeros_like(val)
        smoothed[period-1] = np.mean(val[:period])
        for i in range(period, len(val)):
            smoothed[i] = (smoothed[i-1] * (period-1) + val[i]) / period
        return smoothed
    
    atr = smooth(tr, 14)
    plus_di = 100 * smooth(plus_dm, 14) / (atr + 1e-10)
    minus_di = 100 * smooth(minus_dm, 14) / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = smooth(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses midpoint or ADX < 20
        if position == 1:  # long position
            if close[i] <= donch_mid[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donch_mid[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with ADX filter and volume
            # Bullish breakout: price above Donchian high + ADX > 25 + volume
            if (close[i] > donch_high[i] and 
                adx_aligned[i] > 25 and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Bearish breakout: price below Donchian low + ADX > 25 + volume
            elif (close[i] < donch_low[i] and 
                  adx_aligned[i] > 25 and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Connors RSI (CRSI) with 1d trend filter and volume confirmation
# CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
# Long when CRSI < 10 and price > 1d EMA(200) + volume > 1.5x avg
# Short when CRSI > 90 and price < 1d EMA(200) + volume > 1.5x avg
# Exit when CRSI crosses 50 or price crosses 1d EMA
# Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# CRSI identifies extreme mean reversion; EMA filter ensures trend alignment

name = "6h_crsi_1d_ema_vol_v1"
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
    
    # RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=3, min_periods=3, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=3, min_periods=3, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi3 = 100 - (100 / (1 + rs))
    
    # Streak RSI(2) - consecutive up/down days
    streak = np.zeros_like(close)
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = 0
    streak = np.clip(streak, -2, 2)  # cap at ±2
    # RSI of streak
    streak_delta = np.diff(streak, prepend=streak[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    avg_streak_gain = pd.Series(streak_gain).ewm(span=2, min_periods=2, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=2, min_periods=2, adjust=False).mean().values
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    
    # Percent Rank(100) - where close ranks in last 100 periods
    def percentile_rank(arr, window):
        rank = np.zeros_like(arr)
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            window_data = arr[start:i+1]
            if len(window_data) > 0:
                rank[i] = (np.sum(window_data <= arr[i]) / len(window_data)) * 100
            else:
                rank[i] = 0
        return rank
    percent_rank = percentile_rank(close, 100)
    
    # Connors RSI
    crsi = (rsi3 + rsi_streak + percent_rank) / 3
    
    # 1d EMA(200) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # wait for CRSI components to stabilize
        # Skip if required data not available
        if np.isnan(crsi[i]) or np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: CRSI crosses 50 or price crosses 1d EMA
        if position == 1:  # long position
            if crsi[i] >= 50 or close[i] <= ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if crsi[i] <= 50 or close[i] >= ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for extreme mean reversion with trend filter and volume
            # Long: extremely oversold + above EMA + volume
            if (crsi[i] < 10 and 
                close[i] > ema200_1d_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: extremely overbought + below EMA + volume
            elif (crsi[i] > 90 and 
                  close[i] < ema200_1d_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Tenkan-sen (9-period) and Kijun-sen (26-period) cross
# Senkou Span A and B form the cloud
# Long when TK cross above + price above cloud + price > 1d EMA(50) + volume > 1.5x avg
# Short when TK cross below + price below cloud + price < 1d EMA(50) + volume > 1.5x avg
# Exit when TK cross reverses or price crosses 1d EMA
# Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Ichimoku provides comprehensive trend, momentum, and support/resistance

name = "6h_ichimoku_1d_ema_vol_v1"
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
    tenkan_sen = (pd.Series(high).rolling(window=9, min_periods=9).max().values + 
                  pd.Series(low).rolling(window=9, min_periods=9).min().values) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=26, min_periods=26).max().values + 
                 pd.Series(low).rolling(window=26, min_periods=26).min().values) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high).rolling(window=52, min_periods=52).max().values + 
                      pd.Series(low).rolling(window=52, min_periods=52).min().values) / 2)
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # wait for Ichimoku components to stabilize
        # Skip if required data not available
        if np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or np.isnan(senkou_span_a[i]) or \
           np.isnan(senkou_span_b[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine if price is above or below cloud
        # Cloud top is the higher of Senkou Span A and B
        # Cloud bottom is the lower of Senkou Span A and B
        cloud_top = np.maximum(senkou_span_a[i], senkou_span_b[i])
        cloud_bottom = np.minimum(senkou_span_a[i], senkou_span_b[i])
        
        # Exit conditions: TK cross reverses or price crosses 1d EMA
        if position == 1:  # long position
            if (tenkan_sen[i] <= kijun_sen[i] or  # TK cross down
                close[i] <= ema50_1d_aligned[i]):   # price below EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if (tenkan_sen[i] >= kijun_sen[i] or  # TK cross up
                close[i] >= ema50_1d_aligned[i]):   # price above EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for TK cross with cloud filter, EMA trend, and volume
            # Long: TK cross up + price above cloud + above EMA + volume
            if (tenkan_sen[i] > kijun_sen[i] and  # TK cross up
                tenkan_sen[i-1] <= kijun_sen[i-1] and  # was cross down or equal
                close[i] > cloud_top and            # price above cloud
                close[i] > ema50_1d_aligned[i] and  # price above EMA
                volume[i] > volume_threshold[i]):   # volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + price below cloud + below EMA + volume
            elif (tenkan_sen[i] < kijun_sen[i] and  # TK cross down
                  tenkan_sen[i-1] >= kijun_sen[i-1] and  # was cross up or equal
                  close[i] < cloud_bottom and          # price below cloud
                  close[i] < ema50_1d_aligned[i] and   # price below EMA
                  volume[i] > volume_threshold[i]):    # volume confirmation
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h EMA filter and volume confirmation
# Long when Williams %R < -80 (oversold) + price > 12h EMA + volume > 1.5x avg
# Short when Williams %R > -20 (overbought) + price < 12h EMA + volume > 1.5x avg
# Exit when Williams %R crosses -50 or price crosses 12h EMA
# Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Williams %R identifies extremes; 12h EMA adds trend filter to avoid counter-trend

name = "6h_williamsr_12h_ema_vol_v1"
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
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # 12h EMA (21-period)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Williams %R crosses -50 or price crosses 12h EMA
        if position == 1:  # long position
            if williams_r[i] >= -50 or close[i] <= ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if williams_r[i] <= -50 or close[i] >= ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for mean reversion with EMA filter and volume
            # Long: oversold + above EMA + volume
            if (williams_r[i] < -80 and 
                close[i] > ema_12h_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: overbought + below EMA + volume
            elif (williams_r[i] > -20 and 
                  close[i] < ema_12h_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 12h EMA filter and volume confirmation
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Long when Bull Power > 0 and increasing + price > 12h EMA(50) + volume > 1.5x avg
# Short when Bear Power > 0 and increasing + price < 12h EMA(50) + volume > 1.5x avg
# Exit when power turns negative or price crosses 12h EMA
# Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Elder Ray measures bull/bear strength; 12h EMA adds trend context

name = "6h_elder_ray_12h_ema_vol_v1"
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
    
    # EMA(13) for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span