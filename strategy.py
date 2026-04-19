# US Patent 8,492,727 (C. R. Craig, 2013) - Adaptive Trading System
# Hypothesis: 6h Adaptive RSI with Bollinger Bands and Volume Confirmation
# Uses dynamic RSI thresholds based on Bollinger Band width to adapt to volatility regimes
# In high volatility (wide bands): mean reversion at RSI extremes
# In low volatility (narrow bands): trend following with RSI momentum
# Volume confirmation filters false signals
# Target: 15-25 trades/year per symbol with adaptive regime detection
name = "6h_AdaptiveRSI_BB_Volume"
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
    
    # Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle  # Normalized width
    
    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Bollinger Bands need 20 periods
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width[i]) or np.isnan(rsi[i]) or 
            np.isnan(bb_middle[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
            
        # Regime detection: Bollinger Band width percentile (50-period lookback)
        if i >= 50:
            width_slice = bb_width[i-50:i+1]
            width_percentile = np.percentile(width_slice, 50)  # Median width
            is_high_vol = bb_width[i] > width_percentile * 1.5
        else:
            is_high_vol = False  # Default to low vol regime early on
        
        if position == 0:
            # Entry conditions based on volatility regime
            if is_high_vol:
                # High volatility: mean reversion at RSI extremes
                if rsi[i] < 30 and close[i] < bb_lower[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70 and close[i] > bb_upper[i] and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                # Low volatility: trend following with RSI momentum
                if rsi[i] > 55 and close[i] > bb_middle[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] < 45 and close[i] < bb_middle[i] and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
                    
        elif position == 1:
            # Long exit: RSI overbought OR price touches upper band
            if rsi[i] > 70 or close[i] >= bb_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: RSI oversold OR price touches lower band
            if rsi[i] < 30 or close[i] <= bb_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals