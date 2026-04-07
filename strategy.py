# 6h_ichimoku_cloud_1d_trend_v1
# Hypothesis: 6-hour Ichimoku cloud with 1-day trend filter and volume confirmation
# Uses Ichimoku cloud (Kumo) as dynamic support/resistance and trend filter
# Aligns with 1-day Ichimoku trend for multi-timeframe confirmation
# Volume confirmation ensures momentum behind moves
# Works in bull/bear markets by following higher timeframe trend
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_ichimoku_cloud_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Ichimoku for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    tenkan_sen_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    kijun_sen_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    senkou_span_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2).shift(26)
    senkou_span_b_1d = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                         pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Align 1d Ichimoku to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d.values)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d.values)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d.values)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d.values)
    
    # 6h Ichimoku for entry signals
    tenkan_sen = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    kijun_sen = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = ((pd.Series(high).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Volume confirmation (26-period average)
    vol_ma = pd.Series(volume).rolling(window=26, min_periods=26).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(52, n):
        # Skip if required data not available
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_sen.iloc[i]) or np.isnan(kijun_sen.iloc[i]) or
            np.isnan(senkou_span_a.iloc[i]) or np.isnan(senkou_span_b.iloc[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        # 1d Trend filter: price above/below cloud
        cloud_top_1d = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom_1d = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        price_above_cloud_1d = close[i] > cloud_top_1d
        price_below_cloud_1d = close[i] < cloud_bottom_1d
        
        # 6h Ichimoku signals
        tenkan = tenkan_sen.iloc[i]
        kijun = kijun_sen.iloc[i]
        span_a = senkou_span_a.iloc[i]
        span_b = senkou_span_b.iloc[i]
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # TK cross
        tk_cross_up = tenkan > kijun and tenkan_sen.iloc[i-1] <= kijun_sen.iloc[i-1]
        tk_cross_down = tenkan < kijun and tenkan_sen.iloc[i-1] >= kijun_sen.iloc[i-1]
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit on TK cross down or price below cloud or 1d trend change
            if tk_cross_down or price_below_cloud or not price_above_cloud_1d:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on TK cross up or price above cloud or 1d trend change
            if tk_cross_up or price_above_cloud or not price_below_cloud_1d:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Entry conditions with 1d trend and volume confirmation
            # Long: TK cross up + price above cloud + 1d bullish + volume
            if tk_cross_up and price_above_cloud and price_above_cloud_1d and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short: TK cross down + price below cloud + 1d bearish + volume
            elif tk_cross_down and price_below_cloud and price_below_cloud_1d and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals