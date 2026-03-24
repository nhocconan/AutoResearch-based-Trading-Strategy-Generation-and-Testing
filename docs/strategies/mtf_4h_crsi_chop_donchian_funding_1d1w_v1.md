# Strategy: mtf_4h_crsi_chop_donchian_funding_1d1w_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.014 | +19.6% | -14.0% | 629 | FAIL |
| ETHUSDT | -0.364 | +1.4% | -22.0% | 650 | FAIL |
| SOLUSDT | 0.608 | +79.3% | -18.0% | 685 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.317 | +10.9% | -16.3% | 219 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #681: 4h Primary + 1d/1w HTF — CRSI + Choppiness + Donchian + Funding

Hypothesis: 4h timeframe with daily/weekly HTF filter provides optimal balance between
signal quality and trade frequency. Key innovations:
1. Connors RSI (CRSI) — combines RSI(3) + RSI_Streak(2) + PercentRank(100) for superior mean reversion
2. Choppiness Index regime — CHOP>55=range(mean-revert), CHOP<45=trend(breakout)
3. Donchian(20) breakout for trend entries with HMA confirmation
4. Weekly HMA for macro bias — prevents counter-trend trades
5. Funding Rate Z-score contrarian — proven edge for BTC/ETH in bear markets
6. LOOSE entry thresholds (CRSI<15/>85, not <10/>90) to ensure trade generation
7. Position size 0.25-0.30 with 2.5x ATR trailing stop

Why this should work where others failed:
- CRSI has 75% win rate in research literature for crypto mean reversion
- Dual regime (trend/mean-revert) adapts to market conditions
- 4h TF = ~30-50 trades/year (optimal fee drag vs signal quality)
- Funding contrarian is BEST edge for BTC/ETH in 2022 crash and 2025 bear
- Weekly HTF prevents fighting macro trend

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_donchian_funding_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) — combines 3 components for mean reversion signals.
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    Research shows 75% win rate for CRSI<10 long, CRSI>90 short.
    We use looser thresholds (15/85) to ensure trade generation.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # Component 1: RSI(3) on close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_pad = np.concatenate([[0], gain])
    loss_pad = np.concatenate([[0], loss])
    
    avg_gain = pd.Series(gain_pad).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss_pad).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_close = 100 - (100 / (1 + rs))
    rsi_close = np.clip(rsi_close, 0, 100)
    
    # Component 2: RSI on streak length
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_abs = np.abs(streak)
    streak_gain = np.where(streak > 0, streak_abs, 0)
    streak_loss = np.where(streak < 0, streak_abs, 0)
    
    avg_streak_gain = pd.Series(streak_gain).rolling(window=streak_period, min_periods=streak_period).mean().values
    avg_streak_loss = pd.Series(streak_loss).rolling(window=streak_period, min_periods=streak_period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + rs_streak))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Component 3: PercentRank of daily returns over 100 periods
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = returns[-1]
            rank = np.sum(returns < current_return) / len(returns)
            percent_rank[i] = rank * 100
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — identifies ranging vs trending markets.
    CHOP > 61.8 = choppy/ranging (mean-revert)
    CHOP < 38.2 = trending (trend-follow)
    We use 55/45 thresholds for smoother regime transitions.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high > lowest_low:
            atr_sum = 0.0
            for j in range(i - period + 1, i + 1):
                tr1 = high[j] - low[j]
                tr2 = np.abs(high[j] - close[j - 1]) if j > 0 else tr1
                tr3 = np.abs(low[j] - close[j - 1]) if j > 0 else tr1
                atr_sum += max(tr1, tr2, tr3)
            
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 100
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average — smoother than EMA, less lag."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout levels."""
    n = len(close := high)  # Use high length
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    mid = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        mid[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, mid

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands with %B for overbought/oversold."""
    n = len(close)
    bb_mid = np.full(n, np.nan)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    bb_pct = np.full(n, np.nan)
    
    if n < period:
        return bb_mid, bb_upper, bb_lower, bb_pct
    
    bb_mid = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    bb_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    bb_upper = bb_mid + std_mult * bb_std
    bb_lower = bb_mid - std_mult * bb_std
    
    with np.errstate(divide='ignore', invalid='ignore'):
        bb_pct = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    bb_pct = np.clip(bb_pct, 0, 1)
    return bb_mid, bb_upper, bb_lower, bb_pct

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (4h) indicators
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    bb_mid, bb_upper, bb_lower, bb_pct = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Calculate and align HTF (1d) indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align HTF (1w) indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # === FUNDING RATE CONTRARIAN SIGNAL ===
    funding_zscore = np.zeros(n)
    try:
        import os
        symbol = prices.get('symbol', 'BTCUSDT')
        if isinstance(symbol, (list, np.ndarray)):
            symbol = symbol[0] if len(symbol) > 0 else 'BTCUSDT'
        
        funding_path = f"data/processed/funding/{symbol}.parquet"
        if os.path.exists(funding_path):
            funding_df = pd.read_parquet(funding_path)
            funding_df = funding_df.set_index('open_time').reindex(
                prices['open_time'].values, method='ffill'
            ).fillna(0)
            funding_rate = funding_df['funding_rate'].values
            
            funding_ma = pd.Series(funding_rate).rolling(window=30, min_periods=15).mean().values
            funding_std = pd.Series(funding_rate).rolling(window=30, min_periods=15).std().values
            with np.errstate(divide='ignore', invalid='ignore'):
                funding_zscore = (funding_rate - funding_ma) / (funding_std + 1e-10)
            funding_zscore = np.nan_to_num(funding_zscore, nan=0.0)
    except Exception:
        funding_zscore = np.zeros(n)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start after warmup period for all indicators
        # Skip if indicators not ready
        if np.isnan(crsi_4h[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_value = chop_4h[i]
        is_range_regime = chop_value > 55
        is_trend_regime = chop_value < 45
        
        # === WEEKLY MACRO BIAS (1w HMA) ===
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # === DAILY TREND (1d HMA) ===
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # === CRSI SIGNALS (LOOSE thresholds for trade generation) ===
        crsi_oversold = crsi_4h[i] < 20
        crsi_overbought = crsi_4h[i] > 80
        crsi_extreme_oversold = crsi_4h[i] < 15
        crsi_extreme_overbought = crsi_4h[i] > 85
        
        # === BB POSITION ===
        bb_near_lower = bb_pct[i] < 0.20
        bb_near_upper = bb_pct[i] > 0.80
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === FUNDING CONTRARIAN ===
        funding_extreme_long = funding_zscore[i] > 1.5
        funding_extreme_short = funding_zscore[i] < -1.5
        
        desired_signal = 0.0
        
        # === REGIME 1: TRENDING (CHOP < 45) — Trend Follow with Donchian ===
        if is_trend_regime:
            # Long: Weekly bullish + Daily bullish + Donchian breakout OR CRSI pullback
            if weekly_bullish and daily_bullish:
                if donchian_breakout_long:
                    desired_signal = SIZE_LONG
                elif crsi_oversold and bb_near_lower:
                    desired_signal = SIZE_LONG
            
            # Short: Weekly bearish + Daily bearish + Donchian breakout OR CRSI rally
            elif weekly_bearish and daily_bearish:
                if donchian_breakout_short:
                    desired_signal = -SIZE_SHORT
                elif crsi_overbought and bb_near_upper:
                    desired_signal = -SIZE_SHORT
        
        # === REGIME 2: RANGING (CHOP > 55) — Mean Reversion with CRSI ===
        elif is_range_regime:
            # Long: CRSI extreme oversold + BB near lower + Funding extreme short OR Weekly bullish
            if crsi_extreme_oversold or (crsi_oversold and bb_near_lower):
                if funding_extreme_short or weekly_bullish:
                    desired_signal = SIZE_LONG
            
            # Short: CRSI extreme overbought + BB near upper + Funding extreme long OR Weekly bearish
            if crsi_extreme_overbought or (crsi_overbought and bb_near_upper):
                if funding_extreme_long or weekly_bearish:
                    desired_signal = -SIZE_SHORT
        
        # === REGIME 3: TRANSITION (45 <= CHOP <= 55) — Mixed ===
        else:
            # Use CRSI with trend bias
            if crsi_extreme_oversold and weekly_bullish:
                desired_signal = SIZE_LONG
            elif crsi_extreme_overbought and weekly_bearish:
                desired_signal = -SIZE_SHORT
            elif crsi_oversold and daily_bullish:
                desired_signal = SIZE_LONG * 0.5
            elif crsi_overbought and daily_bearish:
                desired_signal = -SIZE_SHORT * 0.5
        
        # === FUNDING CONTRARIAN OVERRIDE ===
        if funding_extreme_short and crsi_oversold and desired_signal <= 0:
            desired_signal = SIZE_LONG * 0.5
        elif funding_extreme_long and crsi_overbought and desired_signal >= 0:
            desired_signal = -SIZE_SHORT * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                if daily_bullish and crsi_4h[i] < 85:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                if daily_bearish and crsi_4h[i] > 15:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-23 12:38
