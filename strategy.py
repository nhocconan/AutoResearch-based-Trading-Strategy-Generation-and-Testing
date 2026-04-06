#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation.
# Long when price breaks above upper BB after low volatility squeeze (BBW < 20th percentile) during bullish day.
# Short when price breaks below lower BB after squeeze during bearish day.
# Uses daily trend filter to avoid counter-trend trades. Bollinger squeeze identifies low volatility
# periods preceding breakouts, effective in both trending and ranging markets.
# Target: 75-150 total trades over 4 years (19-38/year) to stay within optimal range.

name = "6h_bb_squeeze_breakout_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20-period, 2 std dev)
    close_series = pd.Series(close)
    ma = close_series.rolling(window=20, min_periods=20).mean().values
    std = close_series.rolling(window=20, min_periods=20).std().values
    upper_bb = ma + 2 * std
    lower_bb = ma - 2 * std
    
    # Bollinger Band Width for squeeze detection
    bb_width = (upper_bb - lower_bb) / ma
    # Squeeze threshold: BBW below 20th percentile (low volatility)
    bbw_series = pd.Series(bb_width)
    bbw_percentile = bbw_series.rolling(window=100, min_periods=100).quantile(0.20).values
    squeeze = bb_width < bbw_percentile
    
    # Daily trend filter: bullish/bearish day based on close vs open
    df_1d = get_htf_data(prices, '1d')
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    daily_bullish = daily_close > daily_open  # True for bullish day
    daily_bearish = daily_close < daily_open   # True for bearish day
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if daily trend data not available
        if np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits
        if position == 1:  # long position
            # Exit: price drops below lower BB or daily turn bearish
            if (low[i] <= lower_bb[i] or 
                daily_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above upper BB or daily turn bullish
            if (high[i] >= upper_bb[i] or 
                daily_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation, squeeze, and daily trend filter
            if volume_filter and squeeze[i]:
                # Long: break above upper BB during bullish day
                if (high[i] > upper_bb[i] and 
                    daily_bullish_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: break below lower BB during bearish day
                elif (low[i] < lower_bb[i] and 
                      daily_bearish_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume Weighted Average Price (VWAP) deviation with 1d trend filter and volume confirmation.
# Long when price deviates below VWAP by >1.5σ during bullish day with volume expansion.
# Short when price deviates above VWAP by >1.5σ during bearish day with volume expansion.
# Uses daily trend filter to avoid counter-trend trades. VWAP acts as dynamic support/resistance
# and mean reversion target, effective in both trending and ranging markets.
# Target: 75-150 total trades over 4 years (19-38/year) to stay within optimal range.

name = "6h_vwap_deviation_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWAP calculation (session-based, reset daily)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    
    # VWAP deviation volatility (rolling std of deviation)
    deviation = close - vwap
    dev_series = pd.Series(deviation)
    dev_std = dev_series.rolling(window=50, min_periods=50).std().values
    
    # Daily trend filter: bullish/bearish day based on close vs open
    df_1d = get_htf_data(prices, '1d')
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    daily_bullish = daily_close > daily_open  # True for bullish day
    daily_bearish = daily_close < daily_open   # True for bearish day
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if daily trend data not available
        if np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: price crosses above VWAP or daily turn bearish
            if (close[i] >= vwap[i] or 
                daily_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses below VWAP or daily turn bullish
            if (close[i] <= vwap[i] or 
                daily_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and daily trend filter
            if volume_filter and dev_std[i] > 0:
                z_score = deviation[i] / dev_std[i]
                # Long: price below VWAP by >1.5σ during bullish day
                if (z_score < -1.5 and 
                    daily_bullish_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: price above VWAP by >1.5σ during bearish day
                elif (z_score > 1.5 and 
                      daily_bearish_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Relative Strength Index (RSI) divergence with 1d trend filter and volume confirmation.
# Bullish divergence: price makes lower low, RSI makes higher low during bullish day → long.
# Bearish divergence: price makes higher high, RSI makes lower high during bearish day → short.
# Uses daily trend filter to ensure trades align with higher timeframe momentum.
# RSI divergence identifies weakening momentum and potential reversals, effective in both trending and ranging markets.
# Target: 75-150 total trades over 4 years (19-38/year) to stay within optimal range.

name = "6h_rsi_divergence_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.rolling(window=14, min_periods=14).mean().values
    avg_loss = loss_series.rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily trend filter: bullish/bearish day based on close vs open
    df_1d = get_htf_data(prices, '1d')
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    daily_bullish = daily_close > daily_open  # True for bullish day
    daily_bearish = daily_close < daily_open   # True for bearish day
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Lookback period for divergence detection
    lookback = 10
    
    for i in range(lookback, n):
        # Skip if daily trend data not available
        if np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits
        if position == 1:  # long position
            # Exit: RSI > 70 (overbought) or daily turn bearish
            if (rsi[i] > 70 or 
                daily_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: RSI < 30 (oversold) or daily turn bullish
            if (rsi[i] < 30 or 
                daily_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and daily trend filter
            if volume_filter:
                # Bullish divergence: price lower low, RSI higher low
                if (low[i] < low[i-lookback] and 
                    rsi[i] > rsi[i-lookback] and
                    daily_bullish_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Bearish divergence: price higher high, RSI lower high
                elif (high[i] > high[i-lookback] and 
                      rsi[i] < rsi[i-lookback] and
                      daily_bearish_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Supertrend indicator with 1d trend filter and volume confirmation.
# Long when Supertrend turns bullish during bullish day with volume > 1.5x average.
# Short when Supertrend turns bearish during bearish day with volume confirmation.
# Uses daily trend filter to avoid counter-trend trades. Supertrend combines ATR and trend
# following to capture trends while filtering noise, effective in both trending and ranging markets.
# Target: 75-150 total trades over 4 years (19-38/year) to stay within optimal range.

name = "6h_supertrend_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Supertrend calculation (ATR=10, multiplier=3)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR (10-period)
    tr_series = pd.Series(tr)
    atr = tr_series.rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (3 * atr)
    lower_band = hl2 - (3 * atr)
    
    # Final Upper and Lower Bands
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    for i in range(n):
        if i == 0:
            final_upper[i] = upper_band[i]
            final_lower[i] = lower_band[i]
        else:
            if upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
                final_upper[i] = upper_band[i]
            else:
                final_upper[i] = final_upper[i-1]
            
            if lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
                final_lower[i] = lower_band[i]
            else:
                final_lower[i] = final_lower[i-1]
    
    # Supertrend
    supertrend = np.zeros(n)
    for i in range(n):
        if i == 0:
            supertrend[i] = final_lower[i]
        else:
            if supertrend[i-1] == final_upper[i-1] and close[i] <= final_upper[i]:
                supertrend[i] = final_lower[i]
            elif supertrend[i-1] == final_lower[i-1] and close[i] >= final_lower[i]:
                supertrend[i] = final_upper[i]
            elif supertrend[i-1] == final_upper[i-1]:
                supertrend[i] = final_upper[i]
            else:
                supertrend[i] = final_lower[i]
    
    # Daily trend filter: bullish/bearish day based on close vs open
    df_1d = get_htf_data(prices, '1d')
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    daily_bullish = daily_close > daily_open  # True for bullish day
    daily_bearish = daily_close < daily_open   # True for bearish day
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):
        # Skip if daily trend data not available
        if np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: Supertrend turns bearish or daily turn bearish
            if (supertrend[i] == final_upper[i] or 
                daily_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Supertrend turns bullish or daily turn bullish
            if (supertrend[i] == final_lower[i] or 
                daily_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and daily trend filter
            if volume_filter:
                # Long: Supertrend turns bullish during bullish day
                if (supertrend[i] == final_lower[i] and 
                    supertrend[i-1] == final_upper[i-1] and  # Just turned
                    daily_bullish_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Supertrend turns bearish during bearish day
                elif (supertrend[i] == final_upper[i] and 
                      supertrend[i-1] == final_lower[i-1] and  # Just turned
                      daily_bearish_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Moving Average Convergence Divergence (MACD) histogram reversal with 1d trend filter and volume confirmation.
# Long when MACD histogram crosses above zero during bullish day with volume expansion.
# Short when MACD histogram crosses below zero during bearish day with volume confirmation.
# Uses daily trend filter to avoid counter-trend trades. MACD zero-line cross indicates momentum shift,
# effective in both trending and ranging markets when combined with trend filter.
# Target: 75-150 total trades over 4 years (19-38/year) to stay within optimal range.

name = "6h_macd_histogram_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # MACD calculation (12,26,9)
    close_series = pd.Series(close)
    ema12 = close_series.ewm(span=12, min_periods=12, adjust=False).mean().values
    ema26 = close_series.ewm(span=26, min_periods=26, adjust=False).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, min_periods=9, adjust=False).mean().values
    macd_histogram = macd_line - signal_line
    
    # Daily trend filter: bullish/bearish day based on close vs open
    df_1d = get_htf_data(prices, '1d')
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    daily_bullish = daily_close > daily_open  # True for bullish day
    daily_bearish = daily_close < daily_open   # True for bearish day
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(26, n):
        # Skip if daily trend data not available
        if np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: MACD histogram crosses below zero or daily turn bearish
            if (macd_histogram[i] < 0 and macd_histogram[i-1] >= 0) or \
               daily_bearish_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: MACD histogram crosses above zero or daily turn bullish
            if (macd_histogram[i] > 0 and macd_histogram[i-1] <= 0) or \
               daily_bullish_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and daily trend filter
            if volume_filter:
                # Long: MACD histogram crosses above zero during bullish day
                if (macd_histogram[i] > 0 and macd_histogram[i-1] <= 0 and 
                    daily_bullish_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: MACD histogram crosses below zero during bearish day
                elif (macd_histogram[i] < 0 and macd_histogram[i-1] >= 0 and 
                      daily_bearish_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian channel breakout with 1d trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel during bullish day with volume > 1.5x average.
# Short when price breaks below lower Donchian channel during bearish day with volume confirmation.
# Uses daily trend filter to avoid counter-trend trades. Donchian channels provide clear breakout levels
# that work in both trending and ranging markets when combined with higher timeframe trend.
# Target: 75-150 total trades over 4 years (19-38/year) to stay within optimal range.

name = "6h_donchian20_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily trend filter: bullish/bearish day based on close vs open
    df_1d = get_htf_data(prices, '1d')
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    daily_bullish = daily_close > daily_open  # True for bullish day
    daily_bearish = daily_close < daily_open   # True for bearish day
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if daily trend data not available
        if np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: price drops below lower Donchian or daily turn bearish
            if (low[i] <= lower[i] or 
                daily_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above upper Donchian or daily turn bullish
            if (high[i] >= upper[i] or 
                daily_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and daily trend filter
            if volume_filter:
                # Long: break above upper Donchian during bullish day
                if (high[i] > upper[i] and 
                    daily_bullish_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: break below lower Donchian during bearish day
                elif (low[i] < lower[i] and 
                      daily_bearish_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals

---  END OF FILE ---