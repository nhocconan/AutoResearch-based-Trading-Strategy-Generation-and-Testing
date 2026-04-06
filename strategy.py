#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Ichimoku Cloud Breakout with Daily Trend Filter
# Enter long when: Tenkan crosses above Kijun AND price above Kumo cloud AND 1d trend up (price > 1d EMA50)
# Enter short when: Tenkan crosses below Kijun AND price below Kumo cloud AND 1d trend down (price < 1d EMA50)
# Uses Ichimoku for momentum and trend confirmation, daily EMA for higher timeframe trend filter
# Target: 60-150 trades over 4 years by combining multiple filters to reduce false signals

name = "6h_ichimoku_1d_ema_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Kumo cloud boundaries (shifted forward by 26 periods)
    # For current price, we compare to Senkou spans shifted back 26 periods
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # Fill the first 26 values with NaN (no data)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Kumo top and bottom
    kumo_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    kumo_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Wait for Ichimoku to stabilize
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Tenkan crosses below Kijun OR price below Kumo bottom
            if tenkan[i] < kijun[i] or close[i] < kumo_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Tenkan crosses above Kijun OR price above Kumo top
            if tenkan[i] > kijun[i] or close[i] > kumo_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: TK cross + price in cloud + 1d trend
            # Bullish: Tenkan crosses above Kijun AND price above Kumo AND 1d trend up
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] and
                close[i] > kumo_top[i] and close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Bearish: Tenkan crosses below Kijun AND price below Kumo AND 1d trend down
            elif (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] and
                  close[i] < kumo_bottom[i] and close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted MACD with Daily Trend Filter
# Enter long when: MACD line crosses above signal line AND histogram positive AND 1d trend up (close > EMA50)
# Enter short when: MACD line crosses below signal line AND histogram negative AND 1d trend down (close < EMA50)
# Uses volume-weighted MACD for momentum confirmation and daily EMA for trend filter
# Target: 80-180 trades over 4 years by combining momentum and trend filters

name = "6h_vwmacd_1d_ema_trend_v1"
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
    
    # Volume-weighted MACD calculation
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3
    # Volume-weighted price = typical_price * volume
    vw_price = typical_price * volume
    
    # EMA of volume-weighted price
    vw_price_series = pd.Series(vw_price)
    ema_fast = vw_price_series.ewm(span=12, adjust=False).mean().values
    ema_slow = vw_price_series.ewm(span=26, adjust=False).mean().values
    
    # MACD line
    macd_line = ema_fast - ema_slow
    # Signal line
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
    # Histogram
    macd_hist = macd_line - signal_line
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Wait for MACD to stabilize
        # Skip if required data not available
        if (np.isnan(macd_line[i]) or np.isnan(signal_line[i]) or 
            np.isnan(macd_hist[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: MACD line crosses below signal line OR histogram turns negative
            if macd_line[i] < signal_line[i] or macd_hist[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: MACD line crosses above signal line OR histogram turns positive
            if macd_line[i] > signal_line[i] or macd_hist[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: MACD crossover + histogram sign + 1d trend
            # Bullish: MACD crosses above signal AND histogram positive AND 1d trend up
            if (macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1] and
                macd_hist[i] > 0 and close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Bearish: MACD crosses below signal AND histogram negative AND 1d trend down
            elif (macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1] and
                  macd_hist[i] < 0 and close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX Trend Strength with Daily Volatility Filter
# Enter long when: ADX > 25 AND +DI > -DI AND price > 1d VWAP AND volatility expanding (ATR ratio > 1.0)
# Enter short when: ADX > 25 AND -DI > +DI AND price < 1d VWAP AND volatility expanding (ATR ratio > 1.0)
# Uses ADX for trend strength, DI for direction, 1d VWAP for value, and ATR ratio for volatility regime
# Target: 70-170 trades over 4 years by combining trend, value, and volatility filters

name = "6h_adx_1d_vwap_volatility_v1"
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
    
    # ADX calculation (14 periods)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First value
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # 1d VWAP for value filter
    df_1d = get_htf_data(prices, '1d')
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    volume_1d = df_1d['volume'].values
    vwap_1d = (np.cumsum(typical_price_1d * volume_1d) / np.cumsum(volume_1d))
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volatility filter: ATR ratio (current ATR / 20-period ATR)
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr / atr_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(vwap_1d_aligned[i]) or np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: ADX < 20 OR -DI > +DI OR price < VWAP OR volatility contracting
            if (adx[i] < 20 or minus_di[i] > plus_di[i] or 
                close[i] < vwap_1d_aligned[i] or atr_ratio[i] < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: ADX < 20 OR +DI > -DI OR price > VWAP OR volatility contracting
            if (adx[i] < 20 or plus_di[i] > minus_di[i] or 
                close[i] > vwap_1d_aligned[i] or atr_ratio[i] < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: ADX > 25, DI direction, price vs VWAP, volatility expanding
            if adx[i] > 25 and atr_ratio[i] > 1.0:
                if plus_di[i] > minus_di[i] and close[i] > vwap_1d_aligned[i]:
                    # Strong uptrend, price above value
                    signals[i] = 0.25
                    position = 1
                elif minus_di[i] > plus_di[i] and close[i] < vwap_1d_aligned[i]:
                    # Strong downtrend, price below value
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with Weekly Trend Filter
# Enter long when: Bull Power > 0 AND Bear Power < 0 AND weekly EMA50 up (slope > 0)
# Enter short when: Bear Power < 0 AND Bull Power > 0 AND weekly EMA50 down (slope < 0)
# Uses Elder Ray for bull/bear power balance and weekly EMA for higher timeframe trend
# Target: 60-140 trades over 4 years by combining power balance and trend filters

name = "6h_elder_ray_weekly_trend_v1"
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
    
    # Elder Ray Power (13-period EMA)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Weekly EMA slope for trend direction
    ema_series = pd.Series(ema_50_aligned)
    ema_slope = ema_series.diff().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_slope[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Bull Power <= 0 OR Bear Power >= 0 OR weekly trend down
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or ema_slope[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bear Power >= 0 OR Bull Power <= 0 OR weekly trend up
            if (bear_power[i] >= 0 or bull_power[i] <= 0 or ema_slope[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Power balance + weekly trend
            if bull_power[i] > 0 and bear_power[i] < 0 and ema_slope[i] > 0:
                # Bullish power balance with up trend
                signals[i] = 0.25
                position = 1
            elif bear_power[i] < 0 and bull_power[i] > 0 and ema_slope[i] < 0:
                # Bearish power balance with down trend
                signals[i] = -0.25
                position = -1
    
    return signals

--- End of response ---