#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator + Elder Ray combination with daily trend filter
# Long when Alligator jaws < teeth < lips (bullish alignment) AND Bull Power > 0 (Elder Ray) AND daily close > daily EMA50
# Short when Alligator jaws > teeth > lips (bearish alignment) AND Bear Power < 0 (Elder Ray) AND daily close < daily EMA50
# Exit when alignment breaks or Elder Power reverses
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses Williams Alligator (13,8,5 SMAs) and Elder Ray (EMA13) for trend strength
# Target: 80-120 total trades over 4 years (20-30/year)

name = "6h_alligator_elder_daily_trend_v1"
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
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price = (high + low) / 2
    median_s = pd.Series(median_price)
    jaws = median_s.rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = median_s.rolling(window=8, min_periods=8).mean().values   # Red line
    lips = median_s.rolling(window=5, min_periods=5).mean().values    # Green line
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    ema50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
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
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_daily_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Alligator alignment breaks or Elder Power reverses or daily trend fails
            elif not (jaws[i] < teeth[i] < lips[i]) or bull_power[i] <= 0 or close[i] < ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Alligator alignment breaks or Elder Power reverses or daily trend fails
            elif not (jaws[i] > teeth[i] > lips[i]) or bear_power[i] >= 0 or close[i] > ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Alligator alignment and Elder Power confirmation
            # Bullish: jaws < teeth < lips AND Bull Power > 0 AND daily uptrend
            if (jaws[i] < teeth[i] < lips[i] and
                bull_power[i] > 0 and
                close[i] > ema50_daily_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Bearish: jaws > teeth > lips AND Bear Power < 0 AND daily downtrend
            elif (jaws[i] > teeth[i] > lips[i] and
                  bear_power[i] < 0 and
                  close[i] < ema50_daily_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot levels from daily data with volume confirmation
# Fade at R3/S3 levels (mean reversion), breakout continuation at R4/S4 levels
# Long when price crosses above S3 with volume > 1.3x average AND close > S3
# Short when price crosses below R3 with volume > 1.3x average AND close < R3
# Exit when price reaches opposite S level (S4 for longs, R4 for shorts) or reverses
# Stoploss at 1.5 * ATR(14) to avoid whipsaws
# Position size: 0.28 (28% of capital)
# Uses daily Camarilla calculations (based on previous day's H/L/C)
# Target: 100-160 total trades over 4 years (25-40/year)

name = "6h_camarilla_pivot_v1"
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
    volume = prices['volume'].values
    
    # Daily data for Camarilla calculation (uses previous day's data)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla formulas: 
    # H = previous day high, L = previous day low, C = previous day close
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    prev_high = np.roll(high_daily, 1)
    prev_low = np.roll(low_daily, 1)
    prev_close = np.roll(close_daily, 1)
    # First day has no previous data
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    H_minus_L = prev_high - prev_low
    R4 = prev_close + (H_minus_L * 1.1 / 2)
    R3 = prev_close + (H_minus_L * 1.1 / 4)
    S3 = prev_close - (H_minus_L * 1.1 / 4)
    S4 = prev_close - (H_minus_L * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_daily, R3)
    R4_aligned = align_htf_to_ltf(prices, df_daily, R4)
    S3_aligned = align_htf_to_ltf(prices, df_daily, S3)
    S4_aligned = align_htf_to_ltf(prices, df_daily, S4)
    
    # Volume average for confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(R3_aligned[i]) or np.isnan(R4_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.28
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 1.5 * ATR
            if close[i] < entry_price - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches S4 (target) or reverses below S3
            elif close[i] >= S4_aligned[i] or close[i] < S3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.28
        elif position == -1:  # short position
            # Stoploss: 1.5 * ATR
            if close[i] > entry_price + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches R4 (target) or reverses above R3
            elif close[i] <= R4_aligned[i] or close[i] > R3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.28
        else:
            # Look for entries: fade at S3/R3 with volume confirmation
            # Long: price crosses above S3 with volume spike
            if (close[i] > S3_aligned[i] and 
                close[i-1] <= S3_aligned[i-1] and
                volume[i] > 1.3 * volume_ma[i]):
                signals[i] = 0.28
                position = 1
                entry_price = close[i]
            # Short: price crosses below R3 with volume spike
            elif (close[i] < R3_aligned[i] and 
                  close[i-1] >= R3_aligned[i-1] and
                  volume[i] > 1.3 * volume_ma[i]):
                signals[i] = -0.28
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud system with daily trend filter
# Tenkan-sen (9-period) + Kijun-sen (26-period) cross + Senkou Span A/B cloud
# Long when Tenkan > Kijun AND price > cloud (Senkou Span A) AND daily close > daily EMA50
# Short when Tenkan < Kijun AND price < cloud (Senkou Span B) AND daily close < daily EMA50
# Exit when Tenkan/Kijun cross reverses or price enters cloud
# Stoploss at 2.5 * ATR(14) to account for Ichimoku's slower nature
# Position size: 0.26 (26% of capital)
# Uses Ichimoku components calculated on 6h data with daily EMA filter
# Target: 90-150 total trades over 4 years (22-38/year)

name = "6h_ichimoku_daily_trend_v1"
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
    
    # Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    tenkan = (high_s.rolling(window=9, min_periods=9).max() + 
              low_s.rolling(window=9, min_periods=9).min()) / 2
    tenkan = tenkan.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = (high_s.rolling(window=26, min_periods=26).max() + 
             low_s.rolling(window=26, min_periods=26).min()) / 2
    kijun = kijun.values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods
    senkou_b = ((high_s.rolling(window=52, min_periods=52).max() + 
                 low_s.rolling(window=52, min_periods=52).min()) / 2)
    # Shift both spans forward by 26 periods (for cloud plotting)
    senkou_a = np.roll(senkou_a, 26)
    senkou_b = np.roll(senkou_b, 26)
    # First 26 values are invalid due to shift
    senkou_a[:26] = np.nan
    senkou_b[:26] = np.nan
    
    # Daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    ema50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
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
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema50_daily_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.26
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Tenkan/Kijun cross reverses OR price enters cloud (below Senkou A)
            elif tenkan[i] < kijun[i] or close[i] < senkou_a[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.26
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Tenkan/Kijun cross reverses OR price enters cloud (above Senkou B)
            elif tenkan[i] > kijun[i] or close[i] > senkou_b[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.26
        else:
            # Look for entries: TK cross with price outside cloud and daily trend alignment
            # Bullish: Tenkan > Kijun AND price > Senkou A (above cloud) AND daily uptrend
            if (tenkan[i] > kijun[i] and
                close[i] > senkou_a[i] and
                close[i] > ema50_daily_aligned[i]):
                signals[i] = 0.26
                position = 1
                entry_price = close[i]
            # Bearish: Tenkan < Kijun AND price < Senkou B (below cloud) AND daily downtrend
            elif (tenkan[i] < kijun[i] and
                  close[i] < senkou_b[i] and
                  close[i] < ema50_daily_aligned[i]):
                signals[i] = -0.26
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with daily EMA50 filter and volume spike
# Long when Bull Power > 0 AND close > daily EMA50 AND volume > 2.0x 20-period average
# Short when Bear Power < 0 AND close < daily EMA50 AND volume > 2.0x 20-period average
# Exit when Elder Power reverses or daily trend fails
# Stoploss at 2.0 * ATR(14)
# Position size: 0.27 (27% of capital)
# Uses Elder Ray (EMA13-based) for trend strength measurement
# Target: 110-180 total trades over 4 years (27-45/year)

name = "6h_elder_ray_daily_vol_v1"
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
    volume = prices['volume'].values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    ema50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Volume average for confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_daily_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.27
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bull Power reverses OR daily trend fails
            elif bull_power[i] <= 0 or close[i] < ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.27
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bear Power reverses OR daily trend fails
            elif bear_power[i] >= 0 or close[i] > ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.27
        else:
            # Look for entries: Elder Power alignment with daily trend and volume spike
            # Long: Bull Power > 0, close above daily EMA50, volume spike
            if (bull_power[i] > 0 and
                close[i] > ema50_daily_aligned[i] and
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.27
                position = 1
                entry_price = close[i]
            # Short: Bear Power < 0, close below daily EMA50, volume spike
            elif (bear_power[i] < 0 and
                  close[i] < ema50_daily_aligned[i] and
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.27
                position = -1
                entry_price = close[i]
    
    return signals

---  END OF FILE ---