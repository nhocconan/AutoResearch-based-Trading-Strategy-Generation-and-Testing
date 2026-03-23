# Strategy: mtf_12h_regime_chop_bb_daily_weekly_hma_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.601 | -1.6% | -17.0% | 239 | FAIL |
| ETHUSDT | -0.138 | +12.3% | -15.9% | 259 | FAIL |
| SOLUSDT | 0.568 | +65.6% | -26.5% | 223 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 1.200 | +23.7% | -8.9% | 73 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #161: 12h Regime-Adaptive Strategy with Daily/Weekly HMA Filter
Hypothesis: 12h timeframe captures multi-day swings while avoiding noise. 
Regime detection (Choppiness Index + Bollinger Band Width) switches between
trend-following (CHOP<38.2) and mean-reversion (CHOP>61.8). Daily HMA provides
major trend bias, Weekly HMA confirms macro direction. Entry conditions loosened
to ensure sufficient trades (RSI 30/70 instead of 20/80). ATR stoploss at 2.5*ATR.
This targets the 2022 crash (trend mode) and 2025 consolidation (range mode).
Position sizing: 0.25 entry, 0.125 half-size at 2R profit. Discrete levels minimize fees.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_chop_bb_daily_weekly_hma_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    Reference: E.W. Dreiss
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl > 0, range_hl, 1e-10)
    
    chop = 100 * np.log10(np.sum(atr) / (range_hl * period))
    chop = np.where(np.isnan(chop), 50.0, chop)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_bollinger_bandwidth(close, period=20, std_mult=2.0):
    """
    Calculate Bollinger Band Width for regime detection.
    Low BW = squeeze (potential breakout)
    High BW = expanded (potential mean reversion)
    """
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    std = np.where(std > 0, std, 1e-10)
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bw = (upper - lower) / sma
    bw = np.where(np.isnan(bw), 0.0, bw)
    
    return bw, sma

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    bb_bw, bb_mid = calculate_bollinger_bandwidth(close, 20, 2.0)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    hma_20 = calculate_hma(close, 20)
    hma_50 = calculate_hma(close, 50)
    
    # Calculate BB percentile for regime
    bb_percentile = pd.Series(bb_bw).rolling(window=100, min_periods=50).apply(
        lambda x: np.percentile(x, 50), raw=True
    ).values
    bb_percentile = np.where(np.isnan(bb_percentile), 50.0, bb_percentile)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filters
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # Regime detection
        is_ranging = chop[i] > 55.0  # Loosened from 61.8 for more trades
        is_trending = chop[i] < 45.0  # Loosened from 38.2 for more trades
        bb_expanded = bb_bw[i] > bb_percentile[i]  # Bands expanded
        bb_squeezed = bb_bw[i] < bb_percentile[i]  # Bands squeezed
        
        # 12h trend
        trend_bullish = hma_20[i] > hma_50[i]
        trend_bearish = hma_20[i] < hma_50[i]
        
        # RSI signals (loosened for more trades)
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_rising = rsi[i] > rsi[i-3] if i > 3 else False
        rsi_falling = rsi[i] < rsi[i-3] if i > 3 else False
        
        # MACD signals
        macd_bullish = macd_hist[i] > 0 and macd_hist[i-1] <= 0 if i > 0 else False
        macd_bearish = macd_hist[i] < 0 and macd_hist[i-1] >= 0 if i > 0 else False
        
        new_signal = 0.0
        
        # === MEAN REVERSION MODE (ranging market) ===
        if is_ranging:
            # Long: RSI oversold + price near lower BB + daily not bearish
            if rsi_oversold and close[i] < bb_mid[i] * 0.98:
                if not daily_bearish or rsi_rising:
                    new_signal = SIZE_ENTRY
            
            # Short: RSI overbought + price near upper BB + daily not bullish
            elif rsi_overbought and close[i] > bb_mid[i] * 1.02:
                if not daily_bullish or rsi_falling:
                    new_signal = -SIZE_ENTRY
        
        # === TREND FOLLOWING MODE (trending market) ===
        elif is_trending:
            # Long: HMA crossover + MACD bullish + daily/weekly bullish
            if trend_bullish and hma_20[i-1] <= hma_50[i-1]:
                if macd_bullish or (daily_bullish and weekly_bullish):
                    new_signal = SIZE_ENTRY
            
            # Short: HMA crossover + MACD bearish + daily/weekly bearish
            elif trend_bearish and hma_20[i-1] >= hma_50[i-1]:
                if macd_bearish or (daily_bearish and weekly_bearish):
                    new_signal = -SIZE_ENTRY
            
            # Continuation: trend already established + pullback
            elif trend_bullish and rsi[i] < 50 and rsi_rising:
                if daily_bullish:
                    new_signal = SIZE_ENTRY
            elif trend_bearish and rsi[i] > 50 and rsi_falling:
                if daily_bearish:
                    new_signal = -SIZE_ENTRY
        
        # === BREAKOUT MODE (BB squeeze) ===
        if bb_squeezed and new_signal == 0.0:
            # Breakout long: price breaks above BB mid + volume confirmation
            if close[i] > bb_mid[i] and macd_hist[i] > 0:
                if daily_bullish or weekly_bullish:
                    new_signal = SIZE_ENTRY
            
            # Breakout short: price breaks below BB mid + volume confirmation
            elif close[i] < bb_mid[i] and macd_hist[i] < 0:
                if daily_bearish or weekly_bearish:
                    new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 02:52
