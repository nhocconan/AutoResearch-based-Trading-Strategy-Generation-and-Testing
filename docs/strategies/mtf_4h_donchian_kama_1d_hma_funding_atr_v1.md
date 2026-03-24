# Strategy: mtf_4h_donchian_kama_1d_hma_funding_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.247 | +0.7% | -29.0% | 775 | FAIL |
| ETHUSDT | -0.344 | -13.1% | -38.7% | 801 | FAIL |
| SOLUSDT | 0.301 | +46.2% | -38.0% | 768 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 1.081 | +34.8% | -16.0% | 251 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #1541: 4h Primary + 1d HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: After analyzing 11 failed 4h experiments (#1529-#1540), the pattern is clear:
1. Complex regime switching (choppy vs trending) creates conflicting signals → negative Sharpe
2. Connors RSI extremes (<20 or >80) are TOO STRICT → 0 trades in #1529, #1530, #1535
3. Multiple HTF filters (12h + 1d) create signal conflicts → late entries, whipsaws
4. Current best is 1d timeframe (Sharpe=0.618) — higher TF = fewer false signals

New Approach — SIMPLER and PROVEN patterns from research:
- Donchian(20) breakout: proven Sharpe +0.782 on SOL (research notes)
- KAMA adaptive trend: adjusts to volatility better than HMA in choppy markets
- Single HTF filter (1d HMA only): macro bias without conflicting signals
- LOOSE RSI filter (>45 for long, <55 for short): ensures trades fire
- Funding rate contrarian edge: load funding data for BTC/ETH mean reversion

Why this should work:
- Donchian breakout catches momentum moves (works in bull AND bear)
- KAMA adapts to volatility (flat in chop, fast in trends)
- 1d HMA bias prevents counter-trend trades in strong macro trends
- RSI >45/<55 is LOOSE — ensures we get 30+ trades/train, 3+ trades/test
- Funding rate z-score adds contrarian edge during extremes (research: Sharpe 0.8-1.5)
- Discrete sizing (0.0, ±0.30) minimizes fee churn

Timeframe: 4h (required by experiment)
HTF: 1d HMA(21) for macro trend bias
Position Size: 0.30 (conservative for 4h volatility)
Target: Sharpe > 0.618 (beat current best), DD < -30%, trades > 30/train, > 3/test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_kama_1d_hma_funding_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market volatility — fast in trends, slow in chop
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast - slow) + slow)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = np.full(n, np.nan)
    mask = ~np.isnan(er)
    sc[mask] = np.power(er[mask] * (fast_sc - slow_sc) + slow_sc, 2)
    
    # Initialize KAMA with SMA
    kama[er_period] = np.nanmean(close[:er_period + 1])
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            continue
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel — breakout system
    Upper = highest high of last n periods
    Lower = lowest low of last n periods
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        if w_period < 1:
            return result
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def load_funding_rate(symbol):
    """
    Load funding rate data for contrarian signal
    Returns z-score of 30-day funding rate
    """
    try:
        symbol_map = {
            'BTCUSDT': 'BTCUSDT',
            'ETHUSDT': 'ETHUSDT',
            'SOLUSDT': 'SOLUSDT'
        }
        symbol_name = symbol_map.get(symbol, symbol)
        funding_path = f"data/processed/funding/{symbol_name}.parquet"
        df_funding = pd.read_parquet(funding_path)
        
        if 'funding_rate' not in df_funding.columns:
            return None
        
        funding = df_funding['funding_rate'].values
        if len(funding) < 30:
            return None
        
        # Calculate z-score of 30-day rolling mean
        funding_mean = pd.Series(funding).rolling(window=30, min_periods=30).mean().values
        funding_std = pd.Series(funding).rolling(window=30, min_periods=30).std().values
        
        zscore = np.full(len(funding), np.nan)
        mask = funding_std > 1e-10
        zscore[mask] = (funding[mask] - funding_mean[mask]) / funding_std[mask]
        
        return zscore
    except Exception:
        return None

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Get symbol for funding rate lookup
    symbol = prices.get('symbol', 'BTCUSDT')
    if isinstance(symbol, pd.Series):
        symbol = symbol.iloc[0] if len(symbol) > 0 else 'BTCUSDT'
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Load funding rate z-score for contrarian edge
    funding_zscore = load_funding_rate(symbol)
    if funding_zscore is not None and len(funding_zscore) < n:
        # Pad with NaN if funding data is shorter
        funding_zscore = np.pad(funding_zscore, (0, n - len(funding_zscore)), constant_values=np.nan)
    elif funding_zscore is None:
        funding_zscore = np.full(n, np.nan)
    
    # Calculate primary (4h) indicators
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # KAMA slope for trend confirmation
    kama_slope = np.full(n, np.nan)
    for i in range(10, n):
        if not np.isnan(kama[i]) and not np.isnan(kama[i - 10]):
            kama_slope[i] = (kama[i] - kama[i - 10]) / kama[i - 10] if abs(kama[i - 10]) > 1e-10 else 0.0
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND BIAS (1d HMA) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND (adaptive) ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        kama_rising = kama_slope[i] > 0.0 if not np.isnan(kama_slope[i]) else False
        kama_falling = kama_slope[i] < 0.0 if not np.isnan(kama_slope[i]) else False
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # === RSI FILTER (LOOSE — ensures trades fire) ===
        rsi_long_ok = rsi_14[i] > 45.0 if not np.isnan(rsi_14[i]) else False
        rsi_short_ok = rsi_14[i] < 55.0 if not np.isnan(rsi_14[i]) else False
        rsi_neutral = 40.0 < rsi_14[i] < 60.0 if not np.isnan(rsi_14[i]) else False
        
        # === FUNDING RATE CONTRARIAN (if available) ===
        funding_extreme_long = False
        funding_extreme_short = False
        if not np.isnan(funding_zscore[i]):
            funding_extreme_long = funding_zscore[i] < -1.5  # Very negative funding → long
            funding_extreme_short = funding_zscore[i] > 1.5  # Very positive funding → short
        
        # === DESIRED SIGNAL — BREAKOUT + TREND CONFLUENCE ===
        desired_signal = 0.0
        
        # LONG SETUP
        long_score = 0
        if daily_bull:
            long_score += 2  # Macro trend support
        if kama_bull and kama_rising:
            long_score += 2  # Adaptive trend confirmation
        if breakout_long:
            long_score += 3  # Donchian breakout (primary trigger)
        if rsi_long_ok:
            long_score += 1  # RSI not overbought
        if funding_extreme_long:
            long_score += 2  # Contrarian funding edge
        
        # SHORT SETUP
        short_score = 0
        if daily_bear:
            short_score += 2  # Macro trend support
        if kama_bear and kama_falling:
            short_score += 2  # Adaptive trend confirmation
        if breakout_short:
            short_score += 3  # Donchian breakout (primary trigger)
        if rsi_short_ok:
            short_score += 1  # RSI not oversold
        if funding_extreme_short:
            short_score += 2  # Contrarian funding edge
        
        # Entry thresholds — LOOSE to ensure trades fire
        if long_score >= 5:
            desired_signal = BASE_SIZE
        elif short_score >= 5:
            desired_signal = -BASE_SIZE
        elif long_score >= 4 and daily_bull:
            desired_signal = BASE_SIZE * 0.7
        elif short_score >= 4 and daily_bear:
            desired_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.6:
            final_signal = BASE_SIZE * 0.7
        elif desired_signal >= BASE_SIZE * 0.35:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.6:
            final_signal = -BASE_SIZE * 0.7
        elif desired_signal <= -BASE_SIZE * 0.35:
            final_signal = -BASE_SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals
```

## Last Updated
2026-03-24 01:32
