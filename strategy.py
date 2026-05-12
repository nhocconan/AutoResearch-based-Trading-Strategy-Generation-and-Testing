# 1. Hypothesis: The strategy combines a medium-term trend filter (4h EMA 50) with a volume spike condition and a short-term mean-reversion signal (RSI 2-period) to capture pullbacks in trending markets. The 4h EMA 50 defines the trend direction (bullish if price above EMA, bearish if below), while the RSI 2-period identifies overextended moves within that trend. A volume spike (1.5x 20-period average) confirms the validity of the mean-reversion signal. This approach aims to avoid false signals in ranging markets and whipsaws during strong trends by requiring alignment between trend, momentum, and volume. It is designed to work in both bull and bear markets by following the 4h trend direction. The 4h timeframe balances trade frequency and signal quality, targeting 20-50 trades per year to minimize fee drag. Risk is managed via time-based exits (holding period of 6 bars) to limit exposure and avoid large drawdowns.

# 2. Implementation: The strategy uses the 4h EMA 50 for trend direction, RSI 2-period for mean-reversion signals, and a volume spike filter. The RSI is calculated using Wilder's smoothing (equivalent to EMA with alpha=1/period). The volume spike is defined as volume exceeding 1.5 times the 20-period simple moving average. Entries are taken when the price pulls back to the EMA in the direction of the trend (long when price <= EMA and RSI < 30 in an uptrend; short when price >= EMA and RSI > 70 in a downtrend), confirmed by a volume spike. Exits occur after a fixed holding period of 6 bars to lock in profits and prevent overexposure. Position size is set to 0.25 (25% of capital) to balance risk and return, staying within the 0.40 maximum limit.

# 3. Multi-timeframe analysis is not used in this version to keep the model simple and focused on the 4h timeframe, avoiding the complexity and potential look-ahead bias of multi-timeframe alignment. All indicators are calculated on the 4h data itself, ensuring no look-ahead by using only past and present data up to the current bar.

# 4. Proper min_periods are used in all rolling and EMA calculations to ensure that values are only computed when sufficient data is available, preventing the use of incomplete data.

# 5. The strategy is designed to generate a moderate number of trades (estimated 20-50 per year) by requiring multiple conditions to align (trend, RSI extreme, volume spike), reducing the likelihood of overtrading and fee drag.

# 6. The primary focus is on BTC and ETH, with the expectation that the strategy's logic of trend-following pullbacks with volume confirmation will be effective across these major cryptocurrencies.

# 7. This strategy introduces a novel combination of a medium-term EMA trend filter with a very short-term RSI (2-period) and volume confirmation, which is not commonly seen in the saturated EMA/RSI strategies. This combination aims to capture short-term reversals within a stronger trend, providing an edge in both trending and ranging markets by avoiding trades when the trend is weak or when volume does not confirm the signal.

#!/usr/bin/env python3
"""
4h_EMA50_RSI2_VolumeSpike_Pullback
Hypothesis: Combines 4h EMA 50 trend filter with RSI 2-period mean-reversion and volume spike to capture pullbacks in trending markets. Works in bull/bear by following 4h trend direction.
"""

name = "4h_EMA50_RSI2_VolumeSpike_Pullback"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # 4h EMA 50 for trend
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values

    # RSI 2-period using Wilder's smoothing (EMA with alpha=1/2)
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    # Wilder's smoothing: alpha = 1/period
    ema_up = pd.Series(up).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    ema_down = pd.Series(down).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.where(ema_down != 0, ema_up / ema_down, 0)
    rsi = 100 - (100 / (1 + rs))

    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0

    for i in range(50, n):  # Start after EMA50 warmup
        if np.isnan(ema_50[i]) or np.isnan(rsi[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at or below EMA50 in uptrend, RSI oversold, volume spike
            if (close[i] <= ema_50[i] and 
                close[i] > ema_50[i-1] and  # Ensure uptrend (price above prior EMA)
                rsi[i] < 30 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # SHORT: Price at or above EMA50 in downtrend, RSI overbought, volume spike
            elif (close[i] >= ema_50[i] and 
                  close[i] < ema_50[i-1] and  # Ensure downtrend (price below prior EMA)
                  rsi[i] > 70 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            bars_since_entry += 1
            # Exit after 6 bars to limit exposure
            if bars_since_entry >= 6:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25

    return signals