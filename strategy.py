#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h/1d regime filter
# Long when price below Bollinger lower band (20,2) + RSI<30 + 4h/1d trend up (EMA50 rising)
# Short when price above Bollinger upper band (20,2) + RSI>70 + 4h/1d trend down (EMA50 falling)
# Exit when price crosses Bollinger middle band
# Uses 1h timeframe targeting 60-150 total trades over 4 years (15-37/year)
# Works in both bull and bear by using higher timeframe trend filter to avoid counter-trend trades
# Bollinger Bands provide mean reversion signals, RSI confirms extremity, higher timeframe EMA ensures alignment with larger trend

name = "1h_bollinger_rsi_trend_filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20,2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_middle = sma_20
    
    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_trend_up = ema_4h > np.roll(ema_4h, 1)
    ema_4h_trend_down = ema_4h < np.roll(ema_4h, 1)
    ema_4h_trend_up = align_htf_to_ltf(prices, df_4h, ema_4h_trend_up)
    ema_4h_trend_down = align_htf_to_ltf(prices, df_4h, ema_4h_trend_down)
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_trend_up = ema_1d > np.roll(ema_1d, 1)
    ema_1d_trend_down = ema_1d < np.roll(ema_1d, 1)
    ema_1d_trend_up = align_htf_to_ltf(prices, df_1d, ema_1d_trend_up)
    ema_1d_trend_down = align_htf_to_ltf(prices, df_1d, ema_1d_trend_down)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after EMA warmup
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if required data not available
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(rsi[i]) or \
           np.isnan(ema_4h_trend_up[i]) or np.isnan(ema_4h_trend_down[i]) or \
           np.isnan(ema_1d_trend_up[i]) or np.isnan(ema_1d_trend_down[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses Bollinger middle band
        if position == 1:  # long position
            if close[i] >= bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if close[i] <= bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for mean reversion entries with trend filter
            # Bullish: price below BB lower + RSI oversold + 4h/1d trend up
            if (close[i] < bb_lower[i] and 
                rsi[i] < 30 and 
                ema_4h_trend_up[i] and 
                ema_1d_trend_up[i]):
                signals[i] = 0.20
                position = 1
            # Bearish: price above BB upper + RSI overbought + 4h/1d trend down
            elif (close[i] > bb_upper[i] and 
                  rsi[i] > 70 and 
                  ema_4h_trend_down[i] and 
                  ema_1d_trend_down[i]):
                signals[i] = -0.20
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h breakout strategy with 4h/1d trend filter and volume confirmation
# Long when price breaks above 20-period high + 4h/1d EMA50 up + volume > 1.3x average
# Short when price breaks below 20-period low + 4h/1d EMA50 down + volume > 1.3x average
# Exit when price crosses 20-period EMA
# Uses 1h timeframe targeting 60-150 total trades over 4 years (15-37/year)
# Works in both bull and bear by using higher timeframe trend filter to avoid counter-trend trades
# Breakouts capture momentum, volume confirms conviction, higher timeframe EMA ensures alignment with larger trend

name = "1h_breakout_ema_volume_v1"
timeframe = "1h"
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
    
    # 20-period high/low for breakout
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period EMA for exit
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_trend_up = ema_4h > np.roll(ema_4h, 1)
    ema_4h_trend_down = ema_4h < np.roll(ema_4h, 1)
    ema_4h_trend_up = align_htf_to_ltf(prices, df_4h, ema_4h_trend_up)
    ema_4h_trend_down = align_htf_to_ltf(prices, df_4h, ema_4h_trend_down)
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_trend_up = ema_1d > np.roll(ema_1d, 1)
    ema_1d_trend_down = ema_1d < np.roll(ema_1d, 1)
    ema_1d_trend_up = align_htf_to_ltf(prices, df_1d, ema_1d_trend_up)
    ema_1d_trend_down = align_htf_to_ltf(prices, df_1d, ema_1d_trend_down)
    
    # Volume confirmation: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after EMA warmup
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if required data not available
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_20[i]) or \
           np.isnan(ema_4h_trend_up[i]) or np.isnan(ema_4h_trend_down[i]) or \
           np.isnan(ema_1d_trend_up[i]) or np.isnan(ema_1d_trend_down[i]) or \
           np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses 20-period EMA
        if position == 1:  # long position
            if close[i] <= ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if close[i] >= ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for breakouts with trend and volume confirmation
            # Bullish breakout: price above 20-period high + 4h/1d trend up + volume
            if (close[i] > high_20[i] and 
                ema_4h_trend_up[i] and 
                ema_1d_trend_up[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Bearish breakout: price below 20-period low + 4h/1d trend down + volume
            elif (close[i] < low_20[i] and 
                  ema_4h_trend_down[i] and 
                  ema_1d_trend_down[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h/1d ADX trend filter and Bollinger Band confirmation
# Long when RSI<30 + price touches Bollinger lower band + 4h/1d ADX>25 (trending up)
# Short when RSI>70 + price touches Bollinger upper band + 4h/1d ADX>25 (trending down)
# Exit when RSI crosses 50 (mean reversion complete)
# Uses 1h timeframe targeting 60-150 total trades over 4 years (15-37/year)
# Works in both bull and bear by using ADX to filter for trending conditions only
# RSI identifies overextended conditions, Bollinger Bands provide entry/exit levels, ADX ensures we only trade in trending markets

name = "1h_rsi_bb_adx_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20,2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX components
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # Directional Movement
    dm_plus = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h), 
                       np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)), 
                        np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])  # first average
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smooth = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (tr_smooth + 1e-10)
    di_minus = 100 * dm_minus_smooth / (tr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = wilders_smoothing(dx, period)
    
    # ADX > 25 indicates trending market
    adx_trending = adx > 25
    
    # Determine trend direction using DI+ vs DI-
    adx_trend_up = (di_plus > di_minus) & adx_trending
    adx_trend_down = (di_minus > di_plus) & adx_trending
    
    # Align to 1h timeframe
    adx_trend_up = align_htf_to_ltf(prices, df_4h, adx_trend_up)
    adx_trend_down = align_htf_to_ltf(prices, df_4h, adx_trend_down)
    
    # 1d ADX trend filter (same logic)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    
    # Directional Movement
    dm_plus_1d = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                          np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus_1d = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                           np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth TR, DM+, DM- with Wilder's smoothing
    tr_smooth_1d = wilders_smoothing(tr_1d, period)
    dm_plus_smooth_1d = wilders_smoothing(dm_plus_1d, period)
    dm_minus_smooth_1d = wilders_smoothing(dm_minus_1d, period)
    
    # Directional Indicators
    di_plus_1d = 100 * dm_plus_smooth_1d / (tr_smooth_1d + 1e-10)
    di_minus_1d = 100 * dm_minus_smooth_1d / (tr_smooth_1d + 1e-10)
    
    # DX and ADX
    dx_1d = 100 * np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d + 1e-10)
    adx_1d = wilders_smoothing(dx_1d, period)
    
    # ADX > 25 indicates trending market
    adx_trending_1d = adx_1d > 25
    
    # Determine trend direction using DI+ vs DI-
    adx_trend_up_1d = (di_plus_1d > di_minus_1d) & adx_trending_1d
    adx_trend_down_1d = (di_minus_1d > di_plus_1d) & adx_trending_1d
    
    # Align to 1h timeframe
    adx_trend_up_1d = align_htf_to_ltf(prices, df_1d, adx_trend_up_1d)
    adx_trend_down_1d = align_htf_to_ltf(prices, df_1d, adx_trend_down_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if required data not available
        if np.isnan(rsi[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or \
           np.isnan(adx_trend_up[i]) or np.isnan(adx_trend_down[i]) or \
           np.isnan(adx_trend_up_1d[i]) or np.isnan(adx_trend_down_1d[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: RSI crosses 50 (mean reversion complete)
        if position == 1:  # long position
            if rsi[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if rsi[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for mean reversion entries with trend and BB confirmation
            # Bullish: RSI oversold + price at BB lower + 4h/1d trending up
            if (rsi[i] < 30 and 
                close[i] <= bb_lower[i] and 
                adx_trend_up[i] and 
                adx_trend_up_1d[i]):
                signals[i] = 0.20
                position = 1
            # Bearish: RSI overbought + price at BB upper + 4h/1d trending down
            elif (rsi[i] > 70 and 
                  close[i] >= bb_upper[i] and 
                  adx_trend_down[i] and 
                  adx_trend_down_1d[i]):
                signals[i] = -0.20
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w EMA trend filter and volume confirmation
# Long when price breaks above 20-day high + 1-week EMA50 up + volume > 1.5x average
# Short when price breaks below 20-day low + 1-week EMA50 down + volume > 1.5x average
# Exit when price crosses 10-day EMA or Donchian midpoint reverses
# Uses 1d timeframe targeting 30-100 total trades over 4 years (7-25/year)
# Works in both bull and bear by using higher timeframe trend filter to avoid counter-trend trades
# Donchian captures breakouts, volume confirms conviction, weekly EMA ensures alignment with major trend

name = "1d_donchian_1w_ema_vol_v1"
timeframe = "1d"
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
    
    # 10-period EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # 1-week EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_trend_up = ema_1w > np.roll(ema_1w, 1)
    ema_1w_trend_down = ema_1w < np.roll(ema_1w, 1)
    ema_1w_trend_up = align_htf_to_ltf(prices, df_1w, ema_1w_trend_up)
    ema_1w_trend_down = align_htf_to_ltf(prices, df_1w, ema_1w_trend_down)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after EMA warmup
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_10[i]) or \
           np.isnan(ema_1w_trend_up[i]) or np.isnan(ema_1w_trend_down[i]) or \
           np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses 10-day EMA or Donchian midpoint reverses
        if position == 1:  # long position
            if close[i] <= ema_10[i] or close[i] <= donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if close[i] >= ema_10[i] or close[i] >= donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for breakouts with trend and volume confirmation
            # Bullish breakout: price above Donchian high + 1w EMA up + volume
            if (close[i] > donch_high[i] and 
                ema_1w_trend_up[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Bearish breakout: price below Donchian low + 1w EMA down + volume
            elif (close[i] < donch_low[i] and 
                  ema_1w_trend_down[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals

---  END OF FILE ---