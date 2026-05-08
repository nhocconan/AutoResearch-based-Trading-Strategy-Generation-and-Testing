# 3/4/2025
# Hypothesis: 6h strategy combining 1d Ichimoku cloud filter with 6h Tenkan-Kijun cross and volume confirmation.
# Uses Ichimoku cloud from daily timeframe for trend direction (price above/below cloud),
# Tenkan-Kijun crossover on 6h for entry timing, and volume spike for momentum confirmation.
# Designed for low trade frequency (~20-40/year) to avoid fee drag while capturing trends in both bull and bear markets.
# Ichimoku cloud acts as dynamic support/resistance, reducing whipsaws in ranging markets.

name = "6h_Ichimoku_TK_Cross_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Senkou Span B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    tenkan_sen = (rolling_max(high_1d, 9) + rolling_min(low_1d, 9)) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = (rolling_max(high_1d, 26) + rolling_min(low_1d, 26)) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_span_b = (rolling_max(high_1d, 52) + rolling_min(low_1d, 52)) / 2
    
    # Shift Senkou spans forward by 26 periods
    senkou_span_a_shifted = np.roll(senkou_span_a, 26)
    senkou_span_b_shifted = np.roll(senkou_span_b, 26)
    senkou_span_a_shifted[:26] = np.nan
    senkou_span_b_shifted[:26] = np.nan
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_shifted)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_shifted)
    
    # Get 6h data for volume and TK cross
    # Volume spike: 2x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 2.0)
    
    # TK cross signals
    tk_cross_up = (tenkan_sen_aligned > kijun_sen_aligned) & (tenkan_sen_aligned <= kijun_sen_aligned)
    tk_cross_down = (tenkan_sen_aligned < kijun_sen_aligned) & (tenkan_sen_aligned >= kijun_sen_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud color and position
        green_cloud = senkou_span_a_aligned[i] > senkou_span_b_aligned[i]
        red_cloud = senkou_span_a_aligned[i] < senkou_span_b_aligned[i]
        above_cloud = close[i] > max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        below_cloud = close[i] < min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        in_cloud = not above_cloud and not below_cloud
        
        if position == 0:
            # Enter long: price above green cloud + TK cross up + volume spike
            if above_cloud and green_cloud and tk_cross_up[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price below red cloud + TK cross down + volume spike
            elif below_cloud and red_cloud and tk_cross_down[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below cloud or TK cross down
            if below_cloud or tk_cross_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above cloud or TK cross up
            if above_cloud or tk_cross_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals