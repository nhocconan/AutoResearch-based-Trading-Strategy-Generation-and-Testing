#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour Donchian breakout with 4-hour volume confirmation and daily regime filter
# Long when price breaks above 4h Donchian(20) + volume > 4h SMA(20) volume + 1d volatility regime (BB width > 50th percentile)
# Short when price breaks below 4h Donchian(20) + volume > 4h SMA(20) volume + 1d volatility regime
# Exit when price crosses 4h Donchian midpoint or volatility regime ends
# Stoploss at 2.0 * ATR(14)
# Position size: 0.20 (20% of capital)
# Uses 4h for trend direction, 1h for entry timing, 1d for regime filter
# Target: 100-200 total trades over 4 years (25-50/year)

name = "1h_donchian_4h_vol_1d_regime_v1"
timeframe = "1h"
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
    
    # 4-hour data for Donchian and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4-hour Donchian channels (20)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_4h_s = pd.Series(high_4h)
    low_4h_s = pd.Series(low_4h)
    donchian_high = high_4h_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_4h_s.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 4-hour volume SMA (20)
    vol_4h = df_4h['volume'].values
    vol_4h_s = pd.Series(vol_4h)
    vol_sma = vol_4h_s.rolling(window=20, min_periods=20).mean().values
    
    # Align 4h data to 1h
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    vol_sma_aligned = align_htf_to_ltf(prices, df_4h, vol_sma)
    
    # 1-day data for regime filter (BB width percentile)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    bb_mid = close_1d_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_1d_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate percentile of BB width for regime filter (50th percentile = median)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # 1-hour ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (already datetime64[ms])
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_sma_aligned[i]) or np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(atr[i]) or not (8 <= hours[i] <= 20)):
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
            # Exit: price crosses Donchian midpoint or volatility regime ends
            elif close[i] > donchian_mid_aligned[i] or bb_width_percentile_aligned[i] < 50:
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
            # Exit: price crosses Donchian midpoint or volatility regime ends
            elif close[i] < donchian_mid_aligned[i] or bb_width_percentile_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: Donchian breakout with volume confirmation and volatility regime
            vol_confirm = volume[i] > vol_sma_aligned[i]
            volatile_regime = bb_width_percentile_aligned[i] >= 50
            
            # Long: price breaks above Donchian high + volume confirmation + volatile regime
            if close[i] > donchian_high_aligned[i] and vol_confirm and volatile_regime:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume confirmation + volatile regime
            elif close[i] < donchian_low_aligned[i] and vol_confirm and volatile_regime:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour RSI mean reversion with 4-hour trend filter and daily volatility regime
# Long when RSI(14) < 30 + price > 4h EMA(50) + 1d BB width > 50th percentile (volatile regime)
# Short when RSI(14) > 70 + price < 4h EMA(50) + 1d BB width > 50th percentile
# Exit when RSI crosses 50 or volatility regime ends
# Stoploss at 2.0 * ATR(14)
# Position size: 0.20 (20% of capital)
# Uses 4h for trend direction, 1h for entry timing, 1d for regime filter
# Target: 80-160 total trades over 4 years (20-40/year)

name = "1h_rsi14_4h_ema_1d_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4-hour data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4-hour EMA (50)
    close_4h = df_4h['close'].values
    close_4h_s = pd.Series(close_4h)
    ema_4h = close_4h_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1-day data for regime filter (BB width percentile)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    bb_mid = close_1d_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_1d_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate percentile of BB width for regime filter (50th percentile = median)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # 1-hour RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_s = pd.Series(gain)
    loss_s = pd.Series(loss)
    avg_gain = gain_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1-hour ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (already datetime64[ms])
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(bb_width_percentile_aligned[i]) or np.isnan(atr[i]) or 
            not (8 <= hours[i] <= 20)):
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
            # Exit: RSI crosses 50 or volatility regime ends
            elif rsi[i] >= 50 or bb_width_percentile_aligned[i] < 50:
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
            # Exit: RSI crosses 50 or volatility regime ends
            elif rsi[i] <= 50 or bb_width_percentile_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: RSI extremes with trend filter and volatility regime
            volatile_regime = bb_width_percentile_aligned[i] >= 50
            
            # Long: RSI oversold + price above 4h EMA + volatile regime
            if rsi[i] < 30 and close[i] > ema_4h_aligned[i] and volatile_regime:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: RSI overbought + price below 4h EMA + volatile regime
            elif rsi[i] > 70 and close[i] < ema_4h_aligned[i] and volatile_regime:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour Williams %R with 4-hour ADX trend filter and daily volatility regime
# Long when Williams %R < -80 + ADX(14) > 25 + 1d BB width > 50th percentile (volatile regime)
# Short when Williams %R > -20 + ADX(14) > 25 + 1d BB width > 50th percentile
# Exit when Williams %R crosses -50 or volatility regime ends
# Stoploss at 2.0 * ATR(14)
# Position size: 0.20 (20% of capital)
# Uses 4h for trend strength, 1h for entry timing, 1d for regime filter
# Target: 70-140 total trades over 4 years (18-35/year)

name = "1h_williamsr_4h_adx_1d_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4-hour data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4-hour ADX (14)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h), 
                       np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)), 
                        np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_s = pd.Series(tr)
    dm_plus_s = pd.Series(dm_plus)
    dm_minus_s = pd.Series(dm_minus)
    atr_4h = tr_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = dm_plus_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = dm_minus_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_smooth / (atr_4h + 1e-10)
    minus_di = 100 * dm_minus_smooth / (atr_4h + 1e-10)
    
    # ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    dx_s = pd.Series(dx)
    adx = dx_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # 1-day data for regime filter (BB width percentile)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    bb_mid = close_1d_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_1d_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate percentile of BB width for regime filter (50th percentile = median)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # 1-hour Williams %R (14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # 1-hour ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (already datetime64[ms])
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if required data not available or outside session
        if (np.isnan(wr[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(bb_width_percentile_aligned[i]) or np.isnan(atr[i]) or 
            not (8 <= hours[i] <= 20)):
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
            # Exit: Williams %R crosses -50 or volatility regime ends
            elif wr[i] >= -50 or bb_width_percentile_aligned[i] < 50:
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
            # Exit: Williams %R crosses -50 or volatility regime ends
            elif wr[i] <= -50 or bb_width_percentile_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: Williams %R extremes with trend filter and volatility regime
            strong_trend = adx_aligned[i] > 25
            volatile_regime = bb_width_percentile_aligned[i] >= 50
            
            # Long: Williams %R oversold + strong trend + volatile regime
            if wr[i] < -80 and strong_trend and volatile_regime:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: Williams %R overbought + strong trend + volatile regime
            elif wr[i] > -20 and strong_trend and volatile_regime:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour Stochastic RSI with 4-hour Supertrend filter and daily volatility regime
# Long when StochRSI < 0.2 + Supertrend = uptrend + 1d BB width > 50th percentile (volatile regime)
# Short when StochRSI > 0.8 + Supertrend = downtrend + 1d BB width > 50th percentile
# Exit when StochRSI crosses 0.5 or volatility regime ends
# Stoploss at 2.0 * ATR(14)
# Position size: 0.20 (20% of capital)
# Uses 4h for trend direction, 1h for entry timing, 1d for regime filter
# Target: 60-120 total trades over 4 years (15-30/year)

name = "1h_stochrsi_4h_supertrend_1d_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4-hour data for Supertrend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate 4-hour Supertrend (ATR=10, multiplier=3)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_4h + low_4h) / 2 + 3 * atr_4h
    basic_lb = (high_4h + low_4h) / 2 - 3 * atr_4h
    
    # Final Upper and Lower Bands
    final_ub = np.zeros(len(close_4h))
    final_lb = np.zeros(len(close_4h))
    for i in range(len(close_4h)):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if basic_ub[i] < final_ub[i-1] or close_4h[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
            
            if basic_lb[i] > final_lb[i-1] or close_4h[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend = np.zeros(len(close_4h))
    for i in range(len(close_4h)):
        if i == 0:
            supertrend[i] = final_ub[i]
        else:
            if supertrend[i-1] == final_ub[i-1] and close_4h[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
            elif supertrend[i-1] == final_ub[i-1] and close_4h[i] > final_ub[i]:
                supertrend[i] = final_lb[i]
            elif supertrend[i-1] == final_lb[i-1] and close_4h[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
            elif supertrend[i-1] == final_lb[i-1] and close_4h[i] < final_lb[i]:
                supertrend[i] = final_ub[i]
    
    # Supertrend direction (1 = uptrend, -1 = downtrend)
    supertrend_dir = np.where(close_4h > supertrend, 1, -1)
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_4h, supertrend_dir)
    
    # 1-day data for regime filter (BB width percentile)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    bb_mid = close_1d_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_1d_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate percentile of BB width for regime filter (50th percentile = median)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # 1-hour RSI (14) for StochRSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_s = pd.Series(gain)
    loss_s = pd.Series(loss)
    avg_gain = gain_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic RSI (14,14,3,3)
    rsi_s = pd.Series(rsi)
    rsi_min = rsi_s.rolling(window=14, min_periods=14).min().values
    rsi_max = rsi_s.rolling(window=14, min_periods=14).max().values
    stochrsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
    # Smooth with 3-period SMA
    stochrsi_s = pd.Series(stochrsi)
    stochrsi = stochrsi_s.rolling(window=3, min_periods=3).mean().values
    
    # 1-hour ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (already datetime64[ms])
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if required data not available or outside session
        if (np.isnan(stochrsi[i]) or np.isnan(supertrend_dir_aligned[i]) or 
            np.isnan(bb_width_percentile_aligned[i]) or np.isnan(atr[i]) or 
            not (8 <= hours[i] <= 20)):
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
            # Exit: StochRSI