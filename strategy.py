#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian breakout with weekly pivot direction and volume confirmation
# Long when price breaks above Donchian(20) high AND price > weekly pivot (pivot point) AND volume > 1.5x 20-period avg
# Short when price breaks below Donchian(20) low AND price < weekly pivot AND volume > 1.5x 20-period avg
# Exit on opposite Donchian break or when price crosses below/above weekly pivot
# Uses weekly pivot for trend filter and volume to avoid false breakouts
# Weekly pivot provides long-term trend filter that works in bull/bear markets
# Target: 75-200 trades over 4 years (19-50/year) to balance opportunity and cost

name = "6h_donchian20_weekly_pivot_vol_v1"
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
    
    # Weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Weekly pivot point: (H + L + C) / 3
    pivot_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    pivot_weekly_aligned = align_htf_to_ltf(prices, df_weekly, pivot_weekly)
    
    # 6h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
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
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_weekly_aligned[i]) or np.isnan(donchian_high[i]) or 
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
            # Exit: price breaks below Donchian low or trend turns bearish (below weekly pivot)
            elif close[i] < donchian_low[i] or close[i] < pivot_weekly_aligned[i]:
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
            # Exit: price breaks above Donchian high or trend turns bullish (above weekly pivot)
            elif close[i] > donchian_high[i] or close[i] > pivot_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above Donchian high, price above weekly pivot (bullish trend), volume spike
            if (close[i] > donchian_high[i] and
                close[i] > pivot_weekly_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low, price below weekly pivot (bearish trend), volume spike
            elif (close[i] < donchian_low[i] and
                  close[i] < pivot_weekly_aligned[i] and
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

# Hypothesis: 6-hour Camarilla pivot levels from daily timeframe - fade at R3/S3, breakout continuation at R4/S4
# Long when price breaks above R4 (strong bullish breakout) with volume confirmation
# Short when price breaks below S4 (strong bearish breakdown) with volume confirmation
# Fade trades: Long at S3 with stop at S4, Short at R3 with stop at R4 (mean reversion in range)
# Uses Camarilla levels from daily chart for institutional support/resistance levels
# Volume confirmation reduces false breakouts, works in both trending and ranging markets
# Target: 60-150 trades over 4 years (15-38/year)

name = "6h_camarilla_1d_r4s4_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for Camarilla levels (avoid look-ahead)
    prev_high = df_daily['high'].shift(1).values
    prev_low = df_daily['low'].shift(1).values
    prev_close = df_daily['close'].shift(1).values
    
    # Calculate Camarilla levels for previous day
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    camarilla_r4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s4)
    
    # 6h Donchian channels (20-period) for breakout confirmation
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
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
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
            # Exit: price breaks below S4 (stoploss) or breaks below R3 (take profit on breakout)
            elif close[i] < s4_aligned[i] or close[i] < r3_aligned[i]:
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
            # Exit: price breaks above R4 (stoploss) or breaks above S3 (take profit on breakout)
            elif close[i] > r4_aligned[i] or close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries with volume confirmation
            # Long breakout: price breaks above R4 with volume confirmation
            if (close[i] > r4_aligned[i] and
                close[i] > donchian_high[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short breakdown: price breaks below S4 with volume confirmation
            elif (close[i] < s4_aligned[i] and
                  close[i] < donchian_low[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            # Fade longs at S3 (mean reversion in range)
            elif (close[i] <= s3_aligned[i] and
                  close[i] >= s4_aligned[i] and
                  volume[i] > volume_ma[i]):  # volume confirmation for fade
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Fade shorts at R3 (mean reversion in range)
            elif (close[i] >= r3_aligned[i] and
                  close[i] <= r4_aligned[i] and
                  volume[i] > volume_ma[i]):  # volume confirmation for fade
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud system with daily timeframe filter
# Long when Tenkan-sen > Kijun-sen AND price > Cloud (Senkou Span A/B) AND price > daily Kumo twist
# Short when Tenkan-sen < Kijun-sen AND price < Cloud AND price < daily Kumo twist
# Uses Ichimoku for trend/momentum and daily Kumo twist for higher timeframe trend filter
# Kumo twist (Senkou Span A/B cross) indicates trend change on daily chart
# Works in both bull/bear by following higher timeframe trend
# Target: 50-120 trades over 4 years (13-30/year)

name = "6h_ichimoku_daily_kumo_v1"
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
    
    # Daily data for Kumo twist (Senkou Span A/B cross)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 60:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate Ichimoku components for daily
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_daily = (pd.Series(high_daily).rolling(window=9, min_periods=9).max() + 
                    pd.Series(low_daily).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_daily = (pd.Series(high_daily).rolling(window=26, min_periods=26).max() + 
                   pd.Series(low_daily).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods
    senkou_a_daily = ((tenkan_daily + kijun_daily) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods
    senkou_b_daily = ((pd.Series(high_daily).rolling(window=52, min_periods=52).max() + 
                       pd.Series(low_daily).rolling(window=52, min_periods=52).min()) / 2).shift(52)
    
    # Kumo twist: Senkou Span A crosses Senkou Span B
    # Bullish twist: Senkou A > Senkou B
    # Bearish twist: Senkou A < Senkou B
    kumo_bullish = senkou_a_daily.values > senkou_b_daily.values
    kumo_bullish_aligned = align_htf_to_ltf(prices, df_daily, kumo_bullish.astype(float))
    
    # 6h Ichimoku calculations
    # Tenkan-sen (6-period)
    tenkan_6h = (pd.Series(high).rolling(window=6, min_periods=6).max() + 
                 pd.Series(low).rolling(window=6, min_periods=6).min()) / 2
    # Kijun-sen (26-period)
    kijun_6h = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A: (Tenkan-sen + Kijun-sen)/2 shifted 26 periods
    senkou_a_6h = ((tenkan_6h + kijun_6h) / 2).shift(26)
    # Senkou Span B: (52-period high + 52-period low)/2 shifted 52 periods
    senkou_b_6h = ((pd.Series(high).rolling(window=52, min_periods=52).max() + 
                    pd.Series(low).rolling(window=52, min_periods=52).min()) / 2).shift(52)
    
    # Current Ichimoku values (no shift for current cloud)
    tenkan_val = tenkan_6h.values
    kijun_val = kijun_6h.values
    senkou_a_val = senkou_a_6h.values
    senkou_b_val = senkou_b_6h.values
    
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
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if required data not available
        if (np.isnan(tenkan_val[i]) or np.isnan(kijun_val[i]) or 
            np.isnan(senkou_a_val[i]) or np.isnan(senkou_b_val[i]) or
            np.isnan(kumo_bullish_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine if price is above or below cloud
        # Cloud top = max(Senkou A, Senkou B)
        # Cloud bottom = min(Senkou A, Senkou B)
        cloud_top = np.maximum(senkou_a_val[i], senkou_b_val[i])
        cloud_bottom = np.minimum(senkou_a_val[i], senkou_b_val[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Tenkan crosses below Kijun OR price breaks below cloud OR daily Kumo turns bearish
            elif (tenkan_val[i] < kijun_val[i] or 
                  close[i] < cloud_bottom or 
                  kumo_bullish_aligned[i] < 0.5):  # Bearish twist
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
            # Exit: Tenkan crosses above Kijun OR price breaks above cloud OR daily Kumo turns bullish
            elif (tenkan_val[i] > kijun_val[i] or 
                  close[i] > cloud_top or 
                  kumo_bullish_aligned[i] > 0.5):  # Bullish twist
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Tenkan/Kijun cross with price/Cloud alignment and daily Kumo filter
            # Long: Tenkan > Kijun AND price above cloud AND daily Kumo bullish
            if (tenkan_val[i] > kijun_val[i] and
                price_above_cloud and
                kumo_bullish_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Tenkan < Kijun AND price below cloud AND daily Kumo bearish
            elif (tenkan_val[i] < kijun_val[i] and
                  price_below_cloud and
                  kumo_bullish_aligned[i] < 0.5):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index (Bull Power/Bear Power) with daily EMA filter
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Long when Bull Power > 0 AND Bear Power < 0 AND price > daily EMA(50) AND volume > average
# Short when Bear Power > 0 AND Bull Power < 0 AND price < daily EMA(50) AND volume > average
# Uses Elder Ray to measure bull/bear strength relative to EMA, daily EMA for trend filter
# Works in both bull/bear by following higher timeframe trend
# Volume confirmation reduces false signals
# Target: 60-150 trades over 4 years (15-38/year)

name = "6h_elder_ray_daily_ema50_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for EMA(50) trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    # Daily EMA(50)
    ema_daily = pd.Series(close_daily).ewm(span=50, adjust=False).mean().values
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # 6h EMA(13) for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Elder Ray components
    # Bull Power = High - EMA(13)
    bull_power = high - ema_13
    # Bear Power = EMA(13) - Low
    bear_power = ema_13 - low
    
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
    
    for i in range(13, n):
        # Skip if required data not available
        if (np.isnan(ema_daily_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
            # Exit: Bear Power becomes positive (bears taking control) OR price below daily EMA
            elif bear_power[i] > 0 or close[i] < ema_daily_aligned[i]:
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
            # Exit: Bull Power becomes positive (bulls taking control) OR price above daily EMA
            elif bull_power[i] > 0 or close[i] > ema_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Elder Ray alignment and daily trend filter
            # Long: Bull Power positive (bulls strong) AND Bear Power negative (bears weak) 
            #        AND price above daily EMA (uptrend) AND volume confirmation
            if (bull_power[i] > 0 and
                bear_power[i] < 0 and
                close[i] > ema_daily_aligned[i] and
                volume[i] > volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Bear Power positive (bears strong) AND Bull Power negative (bulls weak)
            #        AND price below daily EMA (downtrend) AND volume confirmation
            elif (bear_power[i] > 0 and
                  bull_power[i] < 0 and
                  close[i] < ema_daily_aligned[i] and
                  volume[i] > volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R with daily EMA(50) and volume filter
# Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when Williams %R crosses above -80 (oversold) AND price > daily EMA(50) AND volume > 1.5x average
# Short when Williams %R crosses below -20 (overbought) AND price < daily EMA(50) AND volume > 1.5x average
# Uses Williams %R for momentum reversals, daily EMA for trend filter, volume for confirmation
# Works in both bull/bear by following higher timeframe trend
# Target: 75-200 trades over 4 years (19-50/year)

name = "6h_williamsr_daily_ema50_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for EMA(50) trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    # Daily EMA(50)
    ema_daily = pd.Series(close_daily).ewm(span=50, adjust=False).mean().values
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Williams %R previous value for crossover detection
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = -50  # neutral
    
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
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(ema_daily_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(williams_r_prev[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
            # Exit: Williams %R crosses below -50 (momentum fading) OR price below daily EMA
            elif (williams_r[i] < -50 and williams_r_prev[i] >= -50) or close[i] < ema_daily_aligned[i]:
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
            # Exit: Williams %R crosses above -50 (momentum fading) OR price above daily EMA
            elif (williams_r[i] > -50 and williams_r_prev[i] <= -50) or close[i] > ema_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for Williams %R crossovers with trend and volume confirmation
            # Long: Williams %R crosses above -80 (oversold bounce) AND price > daily EMA (uptrend) AND volume spike
            if (williams_r[i] > -80 and williams_r_prev[i] <= -80 and
                close[i] > ema_daily_aligned[i] and
                volume[i] > 1