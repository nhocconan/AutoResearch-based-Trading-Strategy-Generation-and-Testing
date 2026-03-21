#!/usr/bin/env python3
"""
EXPERIMENT #014 - Keltner Channel Trend + RSI Pullback + ADX Strength Filter
=============================================================================
Hypothesis: Keltner Channels (ATR-based) adapt better to volatility regimes than 
Bollinger Bands (std-based). Combined with ADX for trend strength confirmation 
and RSI pullback entries, this should reduce false breakouts while capturing 
strong trends. Multi-timeframe: 4h Keltner trend + 1h RSI entries.

Key differences from previous attempts:
- Keltner Channels use ATR (volatility-adaptive) vs Bollinger (std-based)
- ADX(14) > 25 filter ensures we only trade when trend has strength
- ATR trailing stop with 2.0*ATR distance (tighter than Donchian's 2.5)
- Discrete signal levels (0.0, ±0.20, ±0.35) to minimize churn costs

Why this might beat Sharpe=2.931:
- Keltner breakout signals are cleaner in trending markets
- ADX filter avoids choppy sideways periods (major drawdown source)
- Tighter ATR stops preserve capital during reversals
- Multi-timeframe reduces noise while maintaining entry precision
"""

import numpy as np
import pandas as pd

name = "mtf_keltner_rsi_adx_v1"
timeframe = "1h"
leverage = 1.0


def calculate_keltner_channels(high, low, close, period=20, atr_period=10, mult=2.0):
    """
    Calculate Keltner Channels
    Middle = EMA(close, period)
    ATR = ATR(high, low, close, atr_period)
    Upper = Middle + mult * ATR
    Lower = Middle - mult * ATR
    """
    n = len(close)
    
    # Calculate EMA for middle line
    ema = np.zeros(n)
    ema[period - 1] = np.mean(close[:period])
    for i in range(period, n):
        ema[i] = ema[i - 1] + (2.0 / (period + 1)) * (close[i] - ema[i - 1])
    
    # Calculate ATR
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[atr_period - 1] = np.mean(tr[1:atr_period])
    for i in range(atr_period, n):
        atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period
    
    upper = ema + mult * atr
    lower = ema - mult * atr
    
    return upper, lower, ema, atr


def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index)
    Measures trend strength (0-100), not direction
    ADX > 25 indicates strong trend
    """
    n = len(close)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Calculate TR
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    # Smooth with Wilder's method (EMA with alpha = 1/period)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    # Initialize at period
    sum_tr = np.sum(tr[1:period + 1])
    sum_plus_dm = np.sum(plus_dm[1:period + 1])
    sum_minus_dm = np.sum(minus_dm[1:period + 1])
    
    if sum_tr > 0:
        plus_di[period] = 100 * sum_plus_dm / sum_tr
        minus_di[period] = 100 * sum_minus_dm / sum_tr
    
    if plus_di[period] + minus_di[period] > 0:
        dx[period] = 100 * abs(plus_di[period] - minus_di[period]) / (plus_di[period] + minus_di[period])
    
    adx[period] = dx[period]
    
    # Smooth remaining values
    for i in range(period + 1, n):
        # Wilder's smoothing: new = (prev * (n-1) + current) / n
        sum_tr = tr[i] + sum_tr * (period - 1) / period
        sum_plus_dm = plus_dm[i] + sum_plus_dm * (period - 1) / period
        sum_minus_dm = minus_dm[i] + sum_minus_dm * (period - 1) / period
        
        if sum_tr > 0:
            plus_di[i] = 100 * sum_plus_dm / sum_tr
            minus_di[i] = 100 * sum_minus_dm / sum_tr
        
        if plus_di[i] + minus_di[i] > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx, plus_di, minus_di


def calculate_rsi(close, period=14):
    """Calculate RSI with proper initialization"""
    n = len(close)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    # Initialize with SMA
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    
    # Wilder's smoothing
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rsi = np.zeros(n)
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100  # No losses = RSI 100
    
    return rsi


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter"""
    n = len(close)
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        if std > 0:
            zscore[i] = (close[i] - mean) / std
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing and risk
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    
    # 4h Keltner + ADX for trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.ones(n)  # Dummy volume for resample
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Calculate 4h Keltner Channels
    keltner_upper, keltner_lower, keltner_mid, atr_4h = calculate_keltner_channels(
        h_4h, l_4h, c_4h, period=20, atr_period=10, mult=2.0
    )
    
    # Calculate 4h ADX for trend strength
    adx_4h, plus_di_4h, minus_di_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    
    # 4h trend direction based on Keltner position + ADX confirmation
    trend_4h = np.zeros(len(c_4h))
    for i in range(30, len(c_4h)):  # Need 30 bars for ADX to stabilize
        if keltner_upper[i] > keltner_lower[i]:
            price_position = (c_4h[i] - keltner_lower[i]) / (keltner_upper[i] - keltner_lower[i])
            
            # Only trade if ADX shows strong trend
            if adx_4h[i] > 25:
                if price_position > 0.6 and plus_di_4h[i] > minus_di_4h[i]:
                    trend_4h[i] = 1  # Bullish
                elif price_position < 0.4 and minus_di_4h[i] > plus_di_4h[i]:
                    trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = min(idx_1h_to_4h[i], len(trend_4h) - 1)
        if idx_4h >= 0 and idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Map 4h ATR to 1h for stoploss
    atr_4h_mapped = np.zeros(n)
    for i in range(n):
        idx_4h = min(idx_1h_to_4h[i], len(atr_4h) - 1)
        if idx_4h >= 0 and idx_4h < len(atr_4h):
            atr_4h_mapped[i] = atr_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position in good conditions
    SIZE_HALF = 0.20   # Reduced position in marginal conditions
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 55  # Enter short on rally in downtrend
    RSI_EXIT = 50         # Exit when RSI crosses back through midpoint
    
    # Z-score filter to avoid extreme overbought/oversold
    ZSCORE_MAX = 2.5      # Don't enter if price is > 2.5 std from mean
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0   # Tighter stops than Donchian strategy
    
    # ADX minimum for trend confirmation
    ADX_MIN = 25
    
    first_valid = max(80, 30, 14, 20)  # Wait for all indicators
    
    # Track entry prices and stops for trailing stop logic
    long_entry_price = np.full(n, np.nan)
    short_entry_price = np.full(n, np.nan)
    long_stop_price = np.full(n, np.nan)
    short_stop_price = np.full(n, np.nan)
    
    for i in range(first_valid, n):
        # Check for NaN values
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(zscore_1h[i]) or np.isnan(trend_1h[i]):
            signals[i] = 0.0
            continue
        
        # Carry forward previous state if no new signal
        if i > 0:
            signals[i] = signals[i - 1]
            long_entry_price[i] = long_entry_price[i - 1]
            short_entry_price[i] = short_entry_price[i - 1]
            long_stop_price[i] = long_stop_price[i - 1]
            short_stop_price[i] = short_stop_price[i - 1]
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        zscore_val = zscore_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # ATR filter - avoid trading when ATR is extremely high (> 5% of price)
        if atr > 0 and atr / price > 0.05:
            signals[i] = 0.0
            long_entry_price[i] = np.nan
            short_entry_price[i] = np.nan
            continue
        
        # Z-score filter - avoid extreme mean reversion setups
        if abs(zscore_val) > ZSCORE_MAX:
            signals[i] = 0.0
            long_entry_price[i] = np.nan
            short_entry_price[i] = np.nan
            continue
        
        if trend == 1:  # 4h uptrend with ADX confirmation
            # Check stoploss for existing long
            if signals[i] > 0 and not np.isnan(long_stop_price[i]):
                if price < long_stop_price[i]:
                    signals[i] = 0.0
                    long_entry_price[i] = np.nan
                    long_stop_price[i] = np.nan
                    continue
            
            # RSI pullback entry
            if rsi_val < RSI_LONG_ENTRY:
                signals[i] = SIZE_FULL
                long_entry_price[i] = price
                long_stop_price[i] = price - ATR_STOP_MULT * atr
            elif rsi_val < RSI_EXIT and signals[i] == 0:
                # Moderate pullback - half position if not already in
                signals[i] = SIZE_HALF
                long_entry_price[i] = price
                long_stop_price[i] = price - ATR_STOP_MULT * atr
            elif rsi_val > RSI_EXIT and signals[i] > 0:
                # RSI crossed back above 50 - hold or reduce
                if signals[i] == SIZE_HALF:
                    signals[i] = SIZE_FULL  # Upgrade to full position
                # Trail stop higher
                if not np.isnan(long_stop_price[i]):
                    new_stop = price - ATR_STOP_MULT * atr
                    long_stop_price[i] = max(long_stop_price[i], new_stop)
        
        elif trend == -1:  # 4h downtrend with ADX confirmation
            # Check stoploss for existing short
            if signals[i] < 0 and not np.isnan(short_stop_price[i]):
                if price > short_stop_price[i]:
                    signals[i] = 0.0
                    short_entry_price[i] = np.nan
                    short_stop_price[i] = np.nan
                    continue
            
            # RSI rally entry
            if rsi_val > RSI_SHORT_ENTRY:
                signals[i] = -SIZE_FULL
                short_entry_price[i] = price
                short_stop_price[i] = price + ATR_STOP_MULT * atr
            elif rsi_val > RSI_EXIT and signals[i] == 0:
                # Moderate rally - half short if not already in
                signals[i] = -SIZE_HALF
                short_entry_price[i] = price
                short_stop_price[i] = price + ATR_STOP_MULT * atr
            elif rsi_val < RSI_EXIT and signals[i] < 0:
                # RSI crossed back below 50 - hold or reduce
                if signals[i] == -SIZE_HALF:
                    signals[i] = -SIZE_FULL  # Upgrade to full short
                # Trail stop lower
                if not np.isnan(short_stop_price[i]):
                    new_stop = price + ATR_STOP_MULT * atr
                    short_stop_price[i] = min(short_stop_price[i], new_stop)
        
        else:  # No clear trend or ADX too weak
            signals[i] = 0.0
            long_entry_price[i] = np.nan
            short_entry_price[i] = np.nan
            long_stop_price[i] = np.nan
            short_stop_price[i] = np.nan
    
    return signals