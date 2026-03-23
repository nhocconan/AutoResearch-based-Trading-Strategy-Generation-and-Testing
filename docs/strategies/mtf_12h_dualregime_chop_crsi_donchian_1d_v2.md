# Strategy: mtf_12h_dualregime_chop_crsi_donchian_1d_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.547 | +8.2% | -7.1% | 208 | FAIL |
| ETHUSDT | -0.650 | +1.5% | -8.7% | 242 | FAIL |
| SOLUSDT | 0.269 | +35.4% | -21.3% | 210 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.240 | +8.2% | -3.5% | 72 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #356: 12h Primary + 1d HTF — Regime-Adaptive Dual Strategy

Hypothesis: After 350+ experiments, the pattern is clear:
1. 12h timeframe generates optimal trade frequency (20-50/year) - not too many fees, not too few signals
2. Single-regime strategies fail because crypto alternates between trend/chop frequently
3. DUAL REGIME approach: Mean-revert in choppy markets, trend-follow in trending markets
4. 1d HMA(21) for major trend bias, 12h Choppiness Index for regime detection
5. Connors RSI for mean-reversion entries (proven on ETH in exp #352)
6. Donchian breakout for trend-follow entries (proven on SOL in exp #346)
7. Asymmetric sizing: longs 0.25-0.30, shorts 0.15-0.20 (crypto long bias)
8. ATR trailing stop 2.5x to cut losers quickly

Why this might beat current best (Sharpe=0.435):
- Regime detection prevents trend strategies from dying in chop (2022 crash lesson)
- Mean-reversion works in bear/range markets (2025 test period)
- 12h TF avoids 15m/30m fee drag while generating enough signals
- Dual approach ensures trades in ALL market conditions

Position sizing: 0.25-0.30 longs, 0.15-0.20 shorts
Stoploss: 2.5 * ATR trailing
Target: 25-45 trades/year on 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dualregime_chop_crsi_donchian_1d_v2"
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
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/ranging market (mean-revert)
    CHOP < 38.2 = trending market (trend-follow)
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    chop[np.isnan(chop)] = 50.0
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long: CRSI < 10 (oversold)
    Short: CRSI > 90 (overbought)
    """
    close_s = pd.Series(close)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI
    delta = close_s.diff()
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(len(close))
    for i in range(streak_period, len(close)):
        if streak_abs[i] == 0:
            streak_rsi[i] = 50.0
        else:
            # Convert streak to RSI-like value (0-100)
            streak_rsi[i] = 100.0 / (1.0 + streak_abs[i])
            if streak[i] < 0:
                streak_rsi[i] = 100.0 - streak_rsi[i]
    
    # Percent Rank
    percent_rank = np.zeros(len(close))
    for i in range(rank_period, len(close)):
        window = close[i-rank_period:i+1]
        rank = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100.0 * rank / rank_period
    
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    donchian_upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    return donchian_upper, donchian_lower, donchian_mid

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    rsi_14 = calculate_rsi(close, 14)
    hma_12h_21 = calculate_hma(close, period=21)
    hma_12h_8 = calculate_hma(close, period=8)
    sma_200 = calculate_sma(close, 200)
    
    # Donchian channels
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.15
    SHORT_STRONG = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(hma_12h_21[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === 1D MAJOR TREND REGIME (primary direction filter) ===
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME (determines strategy type) ===
        # CHOP > 61.8 = choppy (mean-revert)
        # CHOP < 38.2 = trending (trend-follow)
        # 38.2-61.8 = neutral (use trend-follow with tighter stops)
        choppy_regime = chop_14[i] > 61.8
        trending_regime = chop_14[i] < 38.2
        neutral_regime = not choppy_regime and not trending_regime
        
        # === VOLATILITY REGIME (ATR ratio) ===
        atr_30 = calculate_atr(high, low, close, 30)
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10) if not np.isnan(atr_30[i]) else 1.0
        high_vol = atr_ratio > 1.5
        vol_scale = 0.7 if high_vol else 1.0
        
        # === 12H LOCAL TREND ===
        hma_bullish = hma_12h_8[i] > hma_12h_21[i]
        hma_bearish = hma_12h_8[i] < hma_12h_21[i]
        
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === CONNORS RSI SIGNALS (mean-reversion) ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_neutral = 30.0 < crsi[i] < 70.0
        
        # === ENTRY LOGIC - DUAL REGIME ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # === CHOPPY REGIME: MEAN-REVERSION (Connors RSI) ===
        if choppy_regime:
            # Long: CRSI oversold + bull regime or price > SMA200
            if crsi_oversold:
                if regime_bull or price_above_sma200:
                    new_signal = LONG_BASE * vol_scale
                else:
                    new_signal = LONG_BASE * 0.5 * vol_scale
            
            # Short: CRSI overbought + bear regime or price < SMA200
            if crsi_overbought:
                if regime_bear or not price_above_sma200:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE * vol_scale
                else:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE * 0.5 * vol_scale
        
        # === TRENDING REGIME: TREND-FOLLOW (Donchian Breakout) ===
        elif trending_regime:
            # Long: Donchian breakout + bull regime + HMA bullish
            if donchian_breakout_long and regime_bull and hma_bullish:
                new_signal = LONG_STRONG * vol_scale
            elif donchian_breakout_long and regime_bull:
                new_signal = LONG_BASE * vol_scale
            
            # Short: Donchian breakout + bear regime + HMA bearish
            if donchian_breakout_short and regime_bear and hma_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG * vol_scale
            elif donchian_breakout_short and regime_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
        
        # === NEUTRAL REGIME: HYBRID (both strategies) ===
        elif neutral_regime:
            # Mean-reversion with trend bias
            if crsi_oversold and (regime_bull or hma_bullish):
                new_signal = LONG_BASE * vol_scale
            elif crsi_overbought and (regime_bear or hma_bearish):
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # Trend-follow with tighter requirements
            if donchian_breakout_long and regime_bull:
                if new_signal == 0.0:
                    new_signal = LONG_BASE * 0.8 * vol_scale
            if donchian_breakout_short and regime_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8 * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 12h) ===
        # Force trade if no signal for 12 bars (~6 days on 12h)
        if bars_since_last_trade > 12 and new_signal == 0.0 and not in_position:
            if regime_bull and crsi[i] < 40.0:
                new_signal = LONG_BASE * 0.5 * vol_scale
            elif regime_bear and crsi[i] > 60.0:
                new_signal = -SHORT_BASE * 0.5 * vol_scale
            elif crsi_oversold and price_above_sma200:
                new_signal = LONG_BASE * 0.5 * vol_scale
            elif crsi_overbought and not price_above_sma200:
                new_signal = -SHORT_BASE * 0.5 * vol_scale
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and crsi_overbought:
                crsi_exit = True
            if position_side < 0 and crsi_oversold:
                crsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and regime_bear and close[i] < hma_12h_21[i]:
                regime_reversal = True
            if position_side < 0 and regime_bull and close[i] > hma_12h_21[i]:
                regime_reversal = True
        
        if stoploss_triggered or crsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG * vol_scale
            elif new_signal > 0:
                new_signal = LONG_BASE * vol_scale
            elif new_signal < -0.18:
                new_signal = -SHORT_STRONG * vol_scale
            else:
                new_signal = -SHORT_BASE * vol_scale
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-23 03:10
