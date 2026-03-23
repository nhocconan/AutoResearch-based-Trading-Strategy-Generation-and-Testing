# Strategy: mtf_30m_regime_adx_bb_4h_hma_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -4.036 | -69.3% | -69.4% | 5350 | FAIL |
| ETHUSDT | -0.561 | -1.4% | -13.6% | 209 | FAIL |
| SOLUSDT | 0.315 | +43.1% | -20.0% | 199 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.251 | +8.9% | -7.4% | 68 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #002: 30m Regime-Adaptive Strategy with 4h HMA Trend Filter
Hypothesis: 30m timeframe captures intraday swings while 4h HMA provides trend bias.
Regime detection via ADX + Bollinger Bandwidth: trend-follow in strong trends (ADX>25, BB expanding),
mean-reversion in ranges (ADX<20, BB contracting). This adapts to 2022 crash (trend) and 2025 range.
Key innovation: Dual regime filter reduces false signals. 4h HMA avoids counter-trend entries.
Position sizing: 0.25 base, 0.35 max for strong signals, discrete levels to minimize fee churn.
Stoploss: 2.5*ATR trailing stop to limit drawdown during volatile periods.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_adx_bb_4h_hma_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    
    mask = tr_smooth > 0
    di_plus[mask] = 100 * plus_dm_smooth[mask] / tr_smooth[mask]
    di_minus[mask] = 100 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    dx = np.zeros(n)
    mask2 = (di_plus + di_minus) > 0
    dx[mask2] = 100 * np.abs(di_plus[mask2] - di_minus[mask2]) / (di_plus[mask2] + di_minus[mask2])
    
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[period * 2:] = adx_raw[period * 2:]
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth for regime detection."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    bandwidth[np.isnan(bandwidth)] = 0.0
    
    # Percentile of bandwidth for regime (expanding vs contracting)
    bb_percentile = np.zeros(n := len(close))
    bb_percentile[:] = np.nan
    window = 100
    for i in range(window, n):
        bb_percentile[i] = np.percentile(bandwidth[max(0, i-window):i+1], np.searchsorted(np.sort(bandwidth[max(0, i-window):i+1]), bandwidth[i])) / 100.0
    
    return upper, lower, bandwidth, sma, bb_percentile

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_bandwidth, bb_sma, bb_percentile = calculate_bollinger_bands(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Additional trend filters
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    hma_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.35
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - relaxed for more trades
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # Regime detection via ADX + BB bandwidth
        # Trending: ADX > 25 OR BB expanding (percentile > 0.6)
        # Ranging: ADX < 20 AND BB contracting (percentile < 0.4)
        adx_strong = adx[i] > 25
        adx_weak = adx[i] < 20
        bb_expanding = not np.isnan(bb_percentile[i]) and bb_percentile[i] > 0.6
        bb_contracting = not np.isnan(bb_percentile[i]) and bb_percentile[i] < 0.4
        
        trending_regime = adx_strong or bb_expanding
        ranging_regime = adx_weak and bb_contracting
        
        # Donchian breakout signals (use previous bar to avoid look-ahead)
        breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # Bollinger mean reversion signals
        price_below_lower = close[i] < bb_lower[i] * 1.01
        price_above_upper = close[i] > bb_upper[i] * 0.99
        price_near_lower = close[i] < bb_sma[i] * 0.98
        price_near_upper = close[i] > bb_sma[i] * 1.02
        
        # RSI signals - relaxed thresholds for more trades
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_extreme_oversold = rsi[i] < 30
        rsi_extreme_overbought = rsi[i] > 70
        rsi_rising = rsi[i] > rsi[i - 5] if i >= 5 else False
        rsi_falling = rsi[i] < rsi[i - 5] if i >= 5 else False
        
        # EMA trend confirmation - relaxed
        ema_bullish = close[i] > ema_21[i] and close[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and close[i] < ema_50[i]
        ema_bullish_strong = ema_bullish and close[i] > ema_200[i]
        ema_bearish_strong = ema_bearish and close[i] < ema_200[i]
        
        # HMA crossover signals
        hma_cross_long = hma_21[i] > ema_50[i] and hma_21[i - 1] <= ema_50[i - 1] if i >= 1 else False
        hma_cross_short = hma_21[i] < ema_50[i] and hma_21[i - 1] >= ema_50[i - 1] if i >= 1 else False
        
        new_signal = 0.0
        
        # === TRENDING REGIME: Breakouts and Momentum ===
        if trending_regime:
            # Strong long: breakout + HTF bull + EMA bull + RSI rising
            if breakout_long and bull_trend and ema_bullish and rsi_rising:
                new_signal = SIZE_MAX
            # Strong short: breakout + HTF bear + EMA bear + RSI falling
            elif breakout_short and bear_trend and ema_bearish and rsi_falling:
                new_signal = -SIZE_MAX
            # Moderate long: breakout + HTF bull
            elif breakout_long and bull_trend:
                new_signal = SIZE_BASE
            # Moderate short: breakout + HTF bear
            elif breakout_short and bear_trend:
                new_signal = -SIZE_BASE
            # HMA crossover entries
            elif hma_cross_long and bull_trend:
                new_signal = SIZE_BASE
            elif hma_cross_short and bear_trend:
                new_signal = -SIZE_BASE
        
        # === RANGING REGIME: Mean Reversion ===
        elif ranging_regime:
            # Long at lower BB with RSI oversold
            if price_below_lower and rsi_oversold:
                new_signal = SIZE_BASE
            # Short at upper BB with RSI overbought
            elif price_above_upper and rsi_overbought:
                new_signal = -SIZE_BASE
            # Near BB with extreme RSI
            elif price_near_lower and rsi_extreme_oversold and bull_trend:
                new_signal = SIZE_BASE
            elif price_near_upper and rsi_extreme_overbought and bear_trend:
                new_signal = -SIZE_BASE
        
        # === NEUTRAL REGIME: Conservative entries ===
        else:
            # Only take strongest signals with HTF confirmation
            if breakout_long and bull_trend and rsi_extreme_oversold:
                new_signal = SIZE_BASE
            elif breakout_short and bear_trend and rsi_extreme_overbought:
                new_signal = -SIZE_BASE
            # RSI extreme mean reversion
            elif rsi_extreme_oversold and bull_trend:
                new_signal = SIZE_HALF
            elif rsi_extreme_overbought and bear_trend:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 09:39
