# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: 12h price breaks above Camarilla R1 or below S1 with 1d EMA50 trend filter and volume spike.
# Long when price breaks above R1 AND 1d EMA50 trend up AND 12h volume > 1.5x 20-period average.
# Short when price breaks below S1 AND 1d EMA50 trend down AND 12h volume > 1.5x 20-period average.
# Exit when price re-enters between R1 and S1 (mean reversion) or trend reverses.
# Target: 50-150 total trades over 4 years (12-37/year) for low fee drag.
# Works in bull/bear: Trend filter avoids counter-trend trades; Camarilla levels provide structure in ranging markets.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels using previous day's OHLC
    # Camarilla levels require previous day's high, low, close
    # We'll calculate daily OHLC from 1d data, then shift to avoid look-ahead
    if 'open' not in df_1d.columns:
        # If 1d data doesn't have open, approximate from close (not ideal but fallback)
        prev_close = df_1d['close'].shift(1).values
        # For high/low, we need actual 1d high/low - assume df_1d has them
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close_1d = df_1d['close'].shift(1).values
    else:
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for each 12h bar based on previous day's data
    # Camarilla R1 = C + (H-L) * 1.1/12
    # Camarilla S1 = C - (H-L) * 1.1/12
    hl_range = prev_high - prev_low
    r1 = prev_close + hl_range * 1.1 / 12
    s1 = prev_close - hl_range * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 12h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50 and Camarilla calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R1, volume spike, and 1d EMA50 trending up
            # EMA trending up: current EMA > previous EMA
            ema_trend_up = ema50_1d_aligned[i] > ema50_1d_aligned[i-1]
            long_cond = (close[i] > r1_aligned[i]) and volume_filter[i] and ema_trend_up
            
            # Short conditions: price breaks below S1, volume spike, and 1d EMA50 trending down
            ema_trend_down = ema50_1d_aligned[i] < ema50_1d_aligned[i-1]
            short_cond = (close[i] < s1_aligned[i]) and volume_filter[i] and ema_trend_down
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters below R1 (mean reversion) or trend turns down
            ema_trend_down = ema50_1d_aligned[i] < ema50_1d_aligned[i-1]
            if close[i] < r1_aligned[i] or ema_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters above S1 (mean reversion) or trend turns up
            ema_trend_up = ema50_1d_aligned[i] > ema50_1d_aligned[i-1]
            if close[i] > s1_aligned[i] or ema_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals