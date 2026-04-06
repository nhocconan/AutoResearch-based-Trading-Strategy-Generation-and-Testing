#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14019_6d_ichimoku_1d_trend_follow_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high9 = pd.Series(high).rolling(window=9, min_periods=9).max()
    low9 = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (high9 + low9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high26 = pd.Series(high).rolling(window=26, min_periods=26).max()
    low26 = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (high26 + low26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high52 = pd.Series(high).rolling(window=52, min_periods=52).max()
    low52 = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((high52 + low52) / 2)
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    chikou = pd.Series(close)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values, chikou.values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Ichimoku (once before loop)
    df_1d = get_htf_data(prices, '1d')
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    chikou_aligned = align_htf_to_ltf(prices, df_1d, chikou_1d)
    
    # 6h data for price and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(52, 26, 9, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or \
           np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or \
           np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Ichimoku signals
        # Bullish: Tenkan > Kijun AND price above cloud AND Chikou above price from 26 periods ago
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Chikou condition: compare current close with close 26 periods ago
        chikou_signal = chikou_aligned[i]  # This is the close price from 26 days ago (aligned)
        chikou_above_price = chikou_signal > close[i]
        chikou_below_price = chikou_signal < close[i]
        
        # Tenkan/Kijun cross
        tenkan_above_kijun = tenkan_aligned[i] > kijun_aligned[i]
        tenkan_below_kijun = tenkan_aligned[i] < kijun_aligned[i]
        
        # Generate signals
        if position == 0:
            # Long: bullish TK cross + price above cloud + Chikou above price
            if tenkan_above_kijun and price_above_cloud and chikou_above_price:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            # Short: bearish TK cross + price below cloud + Chikou below price
            elif tenkan_below_kijun and price_below_cloud and chikou_below_price:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or bearish reversal
            if close[i] <= stop_price or (tenkan_below_kijun and price_below_cloud):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or bullish reversal
            if close[i] >= stop_price or (tenkan_above_kijun and price_above_cloud):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14019_6h_alligator_adx_trend_v1"
timeframe = "6h"
leverage = 1.0

def calculate_alligator(high, low, close):
    """Calculate Williams Alligator lines (Jaw, Teeth, Lips)"""
    # Jaw (Blue Line): 13-period SMMA, shifted 8 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    
    # Teeth (Red Line): 8-period SMMA, shifted 5 bars forward
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    
    # Lips (Green Line): 5-period SMMA, shifted 3 bars forward
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    return jaw.values, teeth.values, lips.values

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (alpha = 1/period)
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Handle division by zero
    adx = np.where((di_plus + di_minus) == 0, 0, adx)
    return adx

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for Alligator and ADX (once before loop)
    df_12h = get_htf_data(prices, '12h')
    jaw_12h, teeth_12h, lips_12h = calculate_alligator(
        df_12h['high'].values, df_12h['low'].values, df_12h['close'].values
    )
    adx_12h = calculate_adx(
        df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14
    )
    
    # Align 12h indicators to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 6h data for price and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(13, 8, 5, 14) + 8 + 1  # Jaw shift 8
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or \
           np.isnan(lips_aligned[i]) or np.isnan(adx_aligned[i]) or \
           np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Alligator conditions: Jaw > Teeth > Lips = uptrend, Jaw < Teeth < Lips = downtrend
        alligator_long = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        alligator_short = (jaw_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < lips_aligned[i])
        
        # ADX filter: trend strength > 25
        strong_trend = adx_aligned[i] > 25
        
        # Generate signals
        if position == 0:
            # Long: Alligator uptrend + strong trend
            if alligator_long and strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            # Short: Alligator downtrend + strong trend
            elif alligator_short and strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or trend reversal
            if close[i] <= stop_price or not (alligator_long and strong_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or trend reversal
            if close[i] >= stop_price or not (alligator_short and strong_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14019_6h_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(values, span):
    """Calculate EMA"""
    return pd.Series(values).ewm(span=span, adjust=False, min_periods=span).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Elder Ray and regime detection (once before loop)
    df_1d = get_htf_data(prices, '1d')
    ema_13 = calculate_ema(df_1d['close'].values, 13)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema_13
    bear_power = df_1d['low'].values - ema_13
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    
    # 6h data for price and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = 13 + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or \
           np.isnan(ema_13_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Elder Ray signals
        # Bull Power > 0 and rising = bullish
        bullish = (bull_power_aligned[i] > 0) and (bull_power_aligned[i] > bull_power_aligned[i-1])
        # Bear Power < 0 and falling = bearish
        bearish = (bear_power_aligned[i] < 0) and (bear_power_aligned[i] < bear_power_aligned[i-1])
        
        # Regime filter: EMA13 slope for trend strength
        ema_slope = ema_13_aligned[i] - ema_13_aligned[i-1]
        strong_uptrend = ema_slope > 0
        strong_downtrend = ema_slope < 0
        
        # Generate signals
        if position == 0:
            # Long: Bullish Elder Ray + strong uptrend
            if bullish and strong_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            # Short: Bearish Elder Ray + strong downtrend
            elif bearish and strong_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or bearish reversal
            if close[i] <= stop_price or bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or bullish reversal
            if close[i] >= stop_price or bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14019_6h_adx_williams_r_v1"
timeframe = "6h"
leverage = 1.0

def calculate_williams_r(high, low, close, period):
    """Calculate Williams %R"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero
    wr = np.where((highest_high - lowest_low) == 0, -50, wr)
    return wr.values

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (alpha = 1/period)
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Handle division by zero
    adx = np.where((di_plus + di_minus) == 0, 0, adx)
    return adx

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for Williams %R and ADX (once before loop)
    df_12h = get_htf_data(prices, '12h')
    williams_r_12h = calculate_williams_r(
        df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14
    )
    adx_12h = calculate_adx(
        df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14
    )
    
    # Align 12h indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r_12h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 6h data for price and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = 14 + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or \
           np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Williams %R conditions: oversold < -80, overbought > -20
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # ADX filter: trend strength > 20
        strong_trend = adx_aligned[i] > 20
        
        # Generate signals
        if position == 0:
            # Long: Williams %R oversold + strong trend
            if oversold and strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            # Short: Williams %R overbought + strong trend
            elif overbought and strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or overbought condition
            if close[i] <= stop_price or overbought:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or oversold condition
            if close[i] >= stop_price or oversold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14019_6h_ema_crossover_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(values, span):
    """Calculate EMA"""
    return pd.Series(values).ewm(span=span, adjust=False, min_periods=span).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for EMA and volume average (once before loop)
    df_1d = get_htf_data(prices, '1d')
    ema_50 = calculate_ema(df_1d['close'].values, 50)
    ema_200 = calculate_ema(df_1d['close'].values, 200)
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # 6h data for price, EMA crossover, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h EMA for crossover signals
    ema_20 = calculate_ema(close, 20)
    ema_50_6h = calculate_ema(close, 50)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(200, 50, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or \
           np.isnan(volume_ma_aligned[i]) or np.isnan(ema_20[i]) or \
           np.isnan(ema_50_6h[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # 1d trend filter: price above/below EMAs
        bullish_trend = close[i] > ema_50_aligned[i] and close[i] > ema_200_aligned[i]
        bearish_trend = close[i] < ema_50_aligned[i] and close[i] < ema_200_aligned[i]
        
        # 6h EMA crossover
        ema_cross_up = ema_20[i] > ema_50_6h[i] and ema_20[i-1] <= ema_50_6h[i-1]
        ema_cross_down = ema_20[i] < ema_50_6h[i] and ema_20[i-1] >= ema_50_6h[i-1]
        
        # Volume confirmation: current volume > 1.5x 1d average
        volume_ok = volume[i] > (volume_ma_aligned[i] * 1.5)
        
        # Generate signals
        if position == 0:
            # Long: bullish 1d trend + bullish EMA crossover + volume
            if bullish_trend and ema_cross_up and volume_ok:
                signals[i] = 0.25
                position = 1