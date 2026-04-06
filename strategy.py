#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation
# Long when RSI(14) < 30 and 4h EMA(20) > EMA(50) and volume > 1.5x average
# Short when RSI(14) > 70 and 4h EMA(20) < EMA(50) and volume > 1.5x average
# Exit when RSI returns to neutral (40-60 range) or 4h trend reverses
# Uses 4h for trend direction to reduce trade frequency, 1h for precise entry timing
# Works in both bull/bear markets by fading extremes in the direction of higher timeframe trend

name = "1h_rsi_meanrev_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h trend filter: EMA(20) vs EMA(50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    trend_4h = ema_20_4h - ema_50_4h  # >0 = uptrend, <0 = downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # RSI (14-period) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: RSI returns to neutral range OR 4h trend reverses
        if position == 1:  # long position
            if rsi[i] >= 40 or trend_4h_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if rsi[i] <= 60 or trend_4h_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries in direction of 4h trend with RSI extremes
            # Long: RSI oversold (<30) and 4h uptrend + volume confirmation
            if (rsi[i] < 30 and trend_4h_aligned[i] > 0 and volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought (>70) and 4h downtrend + volume confirmation
            elif (rsi[i] > 70 and trend_4h_aligned[i] < 0 and volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and volume confirmation
# Long when price breaks above 20-period high AND 4h EMA(20) > EMA(50) AND volume > 1.5x average
# Short when price breaks below 20-period low AND 4h EMA(20) < EMA(50) AND volume > 1.5x average
# Exit when price crosses opposite Donchian band OR 4h trend reverses
# Uses 4h for trend direction to reduce trade frequency, 1h for precise entry timing
# Works in both bull/bear markets by trading breakouts in the direction of higher timeframe trend

name = "1h_donchian_breakout_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h trend filter: EMA(20) vs EMA(50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    trend_4h = ema_20_4h - ema_50_4h  # >0 = uptrend, <0 = downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Donchian Channel (20-period) on 1h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses opposite Donchian band OR 4h trend reverses
        if position == 1:  # long position
            if close[i] <= donchian_low[i] or trend_4h_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if close[i] >= donchian_high[i] or trend_4h_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for breakouts in direction of 4h trend with volume confirmation
            # Long: price breaks above Donchian high AND 4h uptrend + volume confirmation
            if (close[i] > donchian_high[i] and trend_4h_aligned[i] > 0 and volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian low AND 4h downtrend + volume confirmation
            elif (close[i] < donchian_low[i] and trend_4h_aligned[i] < 0 and volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R mean reversion with 4h trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) AND 4h EMA(20) > EMA(50) AND volume > 1.5x average
# Short when Williams %R > -20 (overbought) AND 4h EMA(20) < EMA(50) AND volume > 1.5x average
# Exit when Williams %R returns to -50 level OR 4h trend reverses
# Uses 4h for trend direction to reduce trade frequency, 1h for precise entry timing
# Works in both bull/bear markets by fading extremes in the direction of higher timeframe trend

name = "1h_williamsr_meanrev_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h trend filter: EMA(20) vs EMA(50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    trend_4h = ema_20_4h - ema_50_4h  # >0 = uptrend, <0 = downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Williams %R (14-period) on 1h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    williams_r = williams_r.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if required data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Williams %R returns to -50 level OR 4h trend reverses
        if position == 1:  # long position
            if williams_r[i] >= -50 or trend_4h_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if williams_r[i] <= -50 or trend_4h_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries in direction of 4h trend with Williams %R extremes
            # Long: Williams %R oversold (< -80) and 4h uptrend + volume confirmation
            if (williams_r[i] < -80 and trend_4h_aligned[i] > 0 and volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Short: Williams %R overbought (> -20) and 4h downtrend + volume confirmation
            elif (williams_r[i] > -20 and trend_4h_aligned[i] < 0 and volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Stochastic RSI mean reversion with 4h trend filter and volume confirmation
# Long when StochRSI < 0.2 (oversold) AND 4h EMA(20) > EMA(50) AND volume > 1.5x average
# Short when StochRSI > 0.8 (overbought) AND 4h EMA(20) < EMA(50) AND volume > 1.5x average
# Exit when StochRSI returns to 0.5 level OR 4h trend reverses
# Uses 4h for trend direction to reduce trade frequency, 1h for precise entry timing
# Works in both bull/bear markets by fading extremes in the direction of higher timeframe trend

name = "1h_stochrsi_meanrev_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h trend filter: EMA(20) vs EMA(50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    trend_4h = ema_20_4h - ema_50_4h  # >0 = uptrend, <0 = downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # RSI (14-period) for StochRSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # StochRSI (14-period): (RSI - min(RSI)) / (max(RSI) - min(RSI))
    rsi_series = pd.Series(rsi)
    rsi_min = rsi_series.rolling(window=14, min_periods=14).min()
    rsi_max = rsi_series.rolling(window=14, min_periods=14).max()
    stochrsi = (rsi - rsi_min.values) / (rsi_max.values - rsi_min.values + 1e-10)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if required data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(stochrsi[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: StochRSI returns to 0.5 level OR 4h trend reverses
        if position == 1:  # long position
            if stochrsi[i] >= 0.5 or trend_4h_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if stochrsi[i] <= 0.5 or trend_4h_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries in direction of 4h trend with StochRSI extremes
            # Long: StochRSI oversold (< 0.2) and 4h uptrend + volume confirmation
            if (stochrsi[i] < 0.2 and trend_4h_aligned[i] > 0 and volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Short: StochRSI overbought (> 0.8) and 4h downtrend + volume confirmation
            elif (stochrsi[i] > 0.8 and trend_4h_aligned[i] < 0 and volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Commodity Channel Index (CCI) mean reversion with 4h trend filter and volume confirmation
# Long when CCI < -100 (oversold) AND 4h EMA(20) > EMA(50) AND volume > 1.5x average
# Short when CCI > +100 (overbought) AND 4h EMA(20) < EMA(50) AND volume > 1.5x average
# Exit when CCI returns to 0 level OR 4h trend reverses
# Uses 4h for trend direction to reduce trade frequency, 1h for precise entry timing
# Works in both bull/bear markets by fading extremes in the direction of higher timeframe trend

name = "1h_cci_meanrev_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h trend filter: EMA(20) vs EMA(50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    trend_4h = ema_20_4h - ema_50_4h  # >0 = uptrend, <0 = downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Commodity Channel Index (CCI, 20-period)
    typical_price = (high + low + close) / 3.0
    tp_ma = pd.Series(typical_price).rolling(window=20, min_periods=20).mean()
    tp_mean_dev = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci = (typical_price - tp_ma.values) / (0.015 * tp_mean_dev.values + 1e-10)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(cci[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: CCI returns to 0 level OR 4h trend reverses
        if position == 1:  # long position
            if cci[i] >= 0 or trend_4h_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if cci[i] <= 0 or trend_4h_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries in direction of 4h trend with CCI extremes
            # Long: CCI oversold (< -100) and 4h uptrend + volume confirmation
            if (cci[i] < -100 and trend_4h_aligned[i] > 0 and volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Short: CCI overbought (> +100) and 4h downtrend + volume confirmation
            elif (cci[i] > 100 and trend_4h_aligned[i] < 0 and volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Rate of Change (ROC) mean reversion with 4h trend filter and volume confirmation
# Long when ROC(10) < -5% (oversold) AND 4h EMA(20) > EMA(50) AND volume > 1.5x average
# Short when ROC(10) > +5% (overbought) AND 4h EMA(20) < EMA(50) AND volume > 1.5x average
# Exit when ROC(10) returns to 0% level OR 4h trend reverses
# Uses 4h for trend direction to reduce trade frequency, 1h for precise entry timing
# Works in both bull/bear markets by fading extremes in the direction of higher timeframe trend

name = "1h_roc_meanrev_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h trend filter: EMA(20) vs EMA(50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    trend_4h = ema_20_4h - ema_50_4h  # >0 = uptrend, <0 = downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Rate of Change (ROC, 10-period): (close - close[10]) / close[10] * 100
    roc = np.zeros_like(close)
    for i in range(10, n):
        roc[i] = (close[i] - close[i-10]) / close[i-10] * 100
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):
        # Skip if required data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(roc[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: ROC(10) returns to 0% level OR 4h trend reverses
        if position == 1:  # long position
            if roc[i] >= 0 or trend_4h_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if roc[i] <= 0 or trend_4h_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries in direction of 4h trend with ROC extremes
            # Long: ROC oversold (< -5%) and 4h uptrend + volume confirmation
            if (roc[i] < -5 and trend_4h_aligned[i] > 0 and volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Short: ROC overbought (> +5%) and 4h downtrend + volume confirmation
            elif (roc[i] > 5 and trend_4h_aligned[i] < 0 and volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Money Flow Index (MFI) mean reversion with 4h trend filter and volume confirmation
# Long when MFI < 20 (oversold) AND 4h EMA(20) > EMA(50) AND volume > 1.5x average
# Short when MFI > 80 (overbought) AND 4h EMA(20) < EMA(50) AND volume > 1.5x average
# Exit when MFI returns to 50 level OR 4h trend reverses
# Uses 4h for trend direction to reduce trade frequency, 1h for precise entry timing
# Works in both bull/bear markets by fading extremes in the direction of higher timeframe trend

name = "1h_mfi_meanrev_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h trend filter: EMA(20) vs EMA(50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    trend_4h = ema_20_4h - ema_50_4h  # >0 = uptrend, <0 = downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Money Flow Index (MFI, 14-period)
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume
    
    # Calculate positive and negative money flow
    delta_tp = np.diff(typical_price, prepend=typical_price[0])
    positive_flow = np.where(delta_tp > 0, raw_money_flow, 0)
    negative_flow = np.where(delta_tp < 0, raw_money_flow, 0)
    
    # Sum over 14 periods
    positive_mf = pd.Series(positive_flow).rolling(window=14, min_periods=14).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=14, min_periods=14).sum()
    
    # Money Ratio and MFI
    money_ratio = positive_mf.values / (negative_mf.values + 1e-10)
    mfi = 100 - (100 / (1 + money_ratio))
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if required data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(mfi[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: MFI returns to 50 level OR 4h trend reverses
        if position == 1:  # long position
            if mfi[i] >= 50 or trend_4h_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if mfi[i] <= 50 or trend_4h_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries in direction of 4h trend with MFI extremes
            # Long: MFI oversold (< 20) and 4h uptrend + volume confirmation
            if (mfi[i] < 20 and trend_4h_aligned[i] > 0 and volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Short: MFI overbought (> 80) and 4h downtrend + volume confirmation
            elif (mfi[i] > 80 and trend_4h_aligned[i] < 0 and volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position =