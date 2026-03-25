#!/usr/bin/env python3
"""
Experiment #1154: 1d Primary + 1w HTF — HMA Trend + RSI Pullback + Donchian Breakout

Hypothesis: Daily timeframe with weekly bias provides optimal trade frequency (20-50/year)
while capturing major trend moves. Using HMA for trend direction + RSI pullback entries
+ Donchian breakout confirmation creates high-probability setups that work across
BTC/ETH/SOL in both bull and bear markets.

Key innovations:
1. 1w HMA(21) for long-term bias — only trade in weekly trend direction
2. 1d HMA(21) for intermediate trend confirmation
3. RSI(14) pullback entries: Long when RSI 40-55 in uptrend, Short when RSI 45-60 in downtrend
4. Donchian(20) breakout: Price must break 20-day high/low for trend confirmation
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work on 1d:
- Daily bars filter out noise while capturing multi-week swings
- Weekly HMA ensures we're not fighting the major trend
- RSI pullback (not extreme) entries catch trend continuations
- Donchian breakout confirms momentum before entry
- 20-50 trades/year target minimizes fee drag while capturing major moves
- Works in bull (trend follow) and bear (short rallies) markets

Entry conditions (LOOSE to guarantee trades):
- LONG: price>1w_HMA + price>1d_HMA + RSI(14) 40-55 + price>Donchian_high(20)
- SHORT: price<1w_HMA + price<1d_HMA + RSI(14) 45-60 + price<Donchian_low(20)

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_donchian_weekly_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    donchian_high = np.full(n, np.nan, dtype=np.float64)
    donchian_low = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        donchian_high[i] = np.max(high[i-period+1:i+1])
        donchian_low[i] = np.min(low[i-period+1:i+1])
    
    return donchian_high, donchian_low

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    hma_1d = calculate_hma(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_high, donchian_low = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_1d[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (Weekly HMA) ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d TREND (Daily HMA) ===
        daily_bull = close[i] > hma_1d[i]
        daily_bear = close[i] < hma_1d[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_high = close[i] > donchian_high[i-1] if i > 0 else False
        donchian_breakout_low = close[i] < donchian_low[i-1] if i > 0 else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Weekly bull + Daily bull + RSI pullback + Donchian breakout
        if weekly_bull and daily_bull:
            # RSI pullback zone (not oversold, but pulled back in uptrend)
            if 40.0 <= rsi_14[i] <= 58.0:
                if donchian_breakout_high:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: Weekly bear + Daily bear + RSI pullback + Donchian breakout
        elif weekly_bear and daily_bear:
            # RSI pullback zone (not overbought, but rallied in downtrend)
            if 42.0 <= rsi_14[i] <= 60.0:
                if donchian_breakout_low:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals