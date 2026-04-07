#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R with 12-hour trend filter and volume confirmation
# Williams %R measures momentum: values below -80 indicate oversold, above -20 overbought
# Long when Williams %R crosses above -80 (exit oversold) in 12h uptrend with volume confirmation
# Short when Williams %R crosses below -20 (exit overbought) in 12h downtrend with volume confirmation
# Exit when Williams %R crosses opposite threshold (-50 for mean reversion) or stoploss at 2.5 * ATR
# Uses 12h EMA crossover for trend filter to ensure trades align with higher timeframe trend
# Position size: 0.25 (25% of capital)
# Target: 75-200 total trades over 4 years (19-50/year)
# Williams %R is effective in ranging markets and captures reversals in trends, working in both bull and bear markets

name = "6h_williamsr_12h_trend_vol_v1"
timeframe = "6h"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(25) and EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Average volume for volume confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_25_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(atr[i]) or np.isnan(vol_avg[i])):
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
            # Exit: Williams %R crosses below -50 (mean reversion exit)
            elif williams_r[i] < -50:
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
            # Exit: Williams %R crosses above -50 (mean reversion exit)
            elif williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Williams %R conditions for entry
            # Long: Williams %R crosses above -80 (exit oversold) in uptrend with volume
            long_condition = (williams_r[i] > -80) and (williams_r[i-1] <= -80) and \
                             (ema_25_12h_aligned[i] > ema_50_12h_aligned[i]) and \
                             (volume[i] > 1.8 * vol_avg[i])
            # Short: Williams %R crosses below -20 (exit overbought) in downtrend with volume
            short_condition = (williams_r[i] < -20) and (williams_r[i-1] >= -20) and \
                              (ema_25_12h_aligned[i] < ema_50_12h_aligned[i]) and \
                              (volume[i] > 1.8 * vol_avg[i])
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index (Bull Power/Bear Power) with Bollinger Bands mean reversion
# Elder Ray measures bull/bear power relative to EMA: Bull = High - EMA, Bear = EMA - Low
# Long when Bull Power > 0 and Bear Power < 0 with price below BB lower band (oversold in uptrend)
# Short when Bear Power > 0 and Bull Power < 0 with price above BB upper band (overbought in downtrend)
# Uses 12h EMA(20) > EMA(50) for trend filter to ensure alignment with higher timeframe trend
# Exit when price crosses EMA(20) or stoploss at 2.5 * ATR
# Position size: 0.25 (25% of capital)
# Target: 75-200 total trades over 4 years (19-50/year)
# Elder Ray captures trend strength while Bollinger Bands provide mean reversion entries,
# effective in both trending and ranging markets across bull/bear cycles

name = "6h_elder_ray_bb_mr_v1"
timeframe = "6h"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(20) and EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # EMA(20) for Elder Ray and exit
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Average volume for volume confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(ema_20[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(atr[i]) or np.isnan(vol_avg[i])):
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
            # Exit: price crosses below EMA(20)
            elif close[i] < ema_20[i]:
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
            # Exit: price crosses above EMA(20)
            elif close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Trend filter: 12h EMA(20) > EMA(50) for uptrend, < for downtrend
            uptrend = ema_20_12h_aligned[i] > ema_50_12h_aligned[i]
            downtrend = ema_20_12h_aligned[i] < ema_50_12h_aligned[i]
            
            # Volume confirmation: current volume > 1.8 * average volume
            volume_confirm = volume[i] > 1.8 * vol_avg[i]
            
            # Long: Bull Power > 0 and Bear Power < 0 (uptrend) with price below BB lower (oversold)
            long_condition = (bull_power[i] > 0) and (bear_power[i] < 0) and \
                             (close[i] < bb_lower[i]) and uptrend and volume_confirm
            # Short: Bear Power > 0 and Bull Power < 0 (downtrend) with price above BB upper (overbought)
            short_condition = (bear_power[i] > 0) and (bull_power[i] < 0) and \
                              (close[i] > bb_upper[i]) and downtrend and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Stochastic Oscillator with 12-hour trend filter and volume confirmation
# Stochastic compares closing price to price range over a period: %K = (Close - Low)/(High - Low) * 100
# Long when %K crosses above 20 (exit oversold) in 12h uptrend with volume confirmation
# Short when %K crosses below 80 (exit overbought) in 12h downtrend with volume confirmation
# Exit when %K crosses 50 (mean reversion) or stoploss at 2.5 * ATR
# Uses 12h EMA(25) > EMA(50) for trend filter to align with higher timeframe momentum
# Position size: 0.25 (25% of capital)
# Target: 75-200 total trades over 4 years (19-50/year)
# Stochastic is effective for identifying reversals in both trending and ranging markets,
# working across bull/bear cycles when combined with trend and volume filters

name = "6h_stochastic_12h_trend_vol_v1"
timeframe = "6h"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(25) and EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Stochastic Oscillator (14, 3, 3)
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    k_percent = np.where((highest_high - lowest_low) == 0, 50, k_percent)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Average volume for volume confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_25_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(k_percent[i]) or np.isnan(d_percent[i]) or np.isnan(atr[i]) or np.isnan(vol_avg[i])):
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
            # Exit: %K crosses below 50 (mean reversion exit)
            elif k_percent[i] < 50:
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
            # Exit: %K crosses above 50 (mean reversion exit)
            elif k_percent[i] > 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Stochastic conditions for entry (using %K for signals)
            # Long: %K crosses above 20 (exit oversold) in uptrend with volume
            long_condition = (k_percent[i] > 20) and (k_percent[i-1] <= 20) and \
                             (ema_25_12h_aligned[i] > ema_50_12h_aligned[i]) and \
                             (volume[i] > 1.8 * vol_avg[i])
            # Short: %K crosses below 80 (exit overbought) in downtrend with volume
            short_condition = (k_percent[i] < 80) and (k_percent[i-1] >= 80) and \
                              (ema_25_12h_aligned[i] < ema_50_12h_aligned[i]) and \
                              (volume[i] > 1.8 * vol_avg[i])
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian channel breakout with 12-hour ADX trend strength filter and volume confirmation
# Long when price breaks above Donchian upper(20) with 12h ADX > 25 (strong trend) and volume confirmation
# Short when price breaks below Donchian lower(20) with 12h ADX > 25 (strong trend) and volume confirmation
# Exit when price crosses opposite Donchian level or stoploss at 2.5 * ATR
# Uses 12h ADX to ensure trades only occur in strong trending markets, avoiding whipsaws in ranging periods
# Position size: 0.25 (25% of capital)
# Target: 75-200 total trades over 4 years (19-50/year)
# ADX filter ensures we only trade when there is sufficient trend strength, working in both bull and bear markets
# when trends are present, while avoiding choppy markets that cause false breakouts

name = "6h_donchian20_12h_adx_vol_v1"
timeframe = "6h"
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
    
    # 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h ADX (14-period)
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
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_period = 14
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    atr_smooth = pd.Series(tr_12h).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_smooth
    minus_di = 100 * minus_dm_smooth / atr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx_12h = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Average volume for volume confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_avg[i])):
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
            # Exit: price crosses below Donchian lower(20)
            elif close[i] < lowest_low[i]:
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
            # Exit: price crosses above Donchian upper(20)
            elif close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Trend filter: 12h ADX > 25 indicates strong trend
            strong_trend = adx_12h_aligned[i] > 25
            
            # Volume confirmation: current volume > 1.8 * average volume
            volume_confirm = volume[i] > 1.8 * vol_avg[i]
            
            # Long: price breaks above Donchian upper(20) in strong trend with volume
            if close[i] > highest_high[i] and strong_trend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian lower(20) in strong trend with volume
            elif close[i] < lowest_low[i] and strong_trend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Choppiness Index (CHOP) regime filter with Donchian breakout and volume confirmation
# CHOP > 61.8 indicates ranging market (mean reversion), CHOP < 38.2 indicates trending market
# In ranging markets (CHOP > 61.8): fade at Bollinger Bands (2, 2) with volume confirmation
# In trending markets (CHOP < 38.2): breakout Donchian(20) with volume confirmation
# Uses 12h EMA(25) > EMA(50) for trend direction in trending markets
# Exit when CHOP crosses 50 (regime change) or stoploss at 2.5 * ATR
# Position size: 0.25 (25% of capital)
# Target: 75-200 total trades over 4 years (19-50/year)
# CHOP regime filter adapts strategy to market conditions, avoiding false breakouts in ranging markets
# and avoiding mean reversion in strong trends, working in both bull and bear markets

name = "6h_chop_regime_donchian_bb_vol_v1"
timeframe = "6h"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(25) and EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Choppiness Index (14-period)
    atr = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr[0] = high[0] - low[0]
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (highest_high14 - lowest_low14)) / np.log10(14)
    chop = np.where((highest_high14 - lowest_low14) == 0, 50, chop)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 -