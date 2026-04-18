# 6h_Pivot_R1_S1_Breakout_Volume_ATRFilter_V2
# Hypothesis: Use 12h pivot points (R1/S1) as dynamic support/resistance on 6h chart. 
# Breakout above R1 with volume and ATR filter = long signal; breakdown below S1 = short signal.
# Exit when price returns to pivot point (PP) or ATR collapses.
# Works in bull markets (breakouts above resistance) and bear markets (breakdowns below support).
# Pivots from 12h provide structure; volume confirms conviction; ATR filter avoids chop.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

name = "6h_Pivot_R1_S1_Breakout_Volume_ATRFilter_V2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for pivot points and ATR (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pp = (high_12h + low_12h + close_12h) / 3.0
    r1 = 2 * pp - low_12h
    s1 = 2 * pp - high_12h
    
    # Align pivot points to 6h timeframe (wait for 12h bar to close)
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # Calculate 12h ATR (14-period) for volatility filter
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_period = 14
    atr_12h = np.full_like(tr_12h, np.nan)
    if len(tr_12h) >= atr_period:
        atr_12h[atr_period-1] = np.nanmean(tr_12h[:atr_period])
        for i in range(atr_period, len(tr_12h)):
            if not np.isnan(atr_12h[i-1]) and not np.isnan(tr_12h[i]):
                atr_12h[i] = atr_12h[i-1] * (1 - 1/atr_period) + tr_12h[i] * (1/atr_period)
            else:
                atr_12h[i] = np.nan
    
    # ATR multiplier for volatility filter
    atr_mult = 1.5
    atr_threshold_12h = atr_12h * atr_mult
    atr_threshold_aligned = align_htf_to_ltf(prices, df_12h, atr_threshold_12h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(atr_threshold_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Volatility filter: ATR threshold must be positive (sufficient volatility)
        vol_filter = not np.isnan(atr_threshold_aligned[i]) and atr_threshold_aligned[i] > 0
        
        if position == 0:
            # Long: price breaks above R1 AND volume confirmation AND volatility filter
            long_breakout = close[i] > r1_aligned[i]
            if vol_confirm and vol_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume confirmation AND volatility filter
            elif vol_confirm and vol_filter and close[i] < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot point OR ATR drops below threshold (volatility collapse)
            exit_condition = (close[i] <= pp_aligned[i]) or (np.isnan(atr_threshold_aligned[i]) or atr_threshold_aligned[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot point OR ATR drops below threshold (volatility collapse)
            exit_condition = (close[i] >= pp_aligned[i]) or (np.isnan(atr_threshold_aligned[i]) or atr_threshold_aligned[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals