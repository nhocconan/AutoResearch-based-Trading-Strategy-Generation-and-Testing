# 12h_Camarilla_R1S1_Breakout_1dTrend_Volume
# Hypothesis: 12h Camarilla pivot level R1/S1 breakout with volume confirmation and 1d trend filter.
# Uses Camarilla pivot levels from daily data for precise entry points, confirmed by volume spikes and 1d EMA trend.
# Designed to work in both bull and bear markets by following the 1d trend direction.
# Target: 15-30 trades/year per symbol to avoid excessive fee drag on 12h timeframe.
name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d trend filter: 34-period EMA on close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla pivot levels from previous day (R1/S1)
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    pivot = (high_prev + low_prev + close_prev) / 3
    range_ = high_prev - low_prev
    S1 = close_prev - 1.1 * range_ / 6
    R1 = close_prev + 1.1 * range_ / 6
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    
    # 12h volume average for spike detection
    vol_ema_12h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_12h > 0, volume / vol_ema_12h, 1.0) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for EMA and pivot calculation
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume spike in uptrend
            long_condition = (close[i] > R1_aligned[i]) and vol_spike[i] and uptrend
            # Short breakdown: price breaks below S1 with volume spike in downtrend
            short_condition = (close[i] < S1_aligned[i]) and vol_spike[i] and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price re-enters below R1 or trend turns down
            if (close[i] < R1_aligned[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price re-enters above S1 or trend turns up
            if (close[i] > S1_aligned[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals