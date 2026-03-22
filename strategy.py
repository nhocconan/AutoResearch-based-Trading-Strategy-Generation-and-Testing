#!/usr/bin/env python3
"""
Experiment #360: 1d Regime-Adaptive Strategy with Weekly HMA Bias

Hypothesis: After 359 experiments, the clearest pattern is that STATIC strategies fail
because BTC/ETH markets cycle through distinct regimes (bull/bear/range). For 1d timeframe:

1. REGIME DETECTION (dual-filter):
   - ADX(14) > 25 = TRENDING regime (use breakout/trend-follow entries)
   - ADX(14) < 20 + BB Width < 20th percentile = RANGE regime (use mean-reversion)
   - This adapts to market conditions instead of forcing one logic

2. WEEKLY HMA BIAS (1w HTF via mtf_data):
   - Long bias only when price > 1w HMA(21)
   - Short bias only when price < 1w HMA(21)
   - Filters 50%+ of counter-trend trades that fail in 2022 crash

3. TRENDING REGIME ENTRIES:
   - Long: Price breaks Donchian(20) high + weekly bullish + ADX > 25
   - Short: Price breaks Donchian(20) low + weekly bearish + ADX > 25
   - Captures major moves while avoiding chop

4. RANGING REGIME ENTRIES:
   - Long: RSI(7) < 25 + price < BB lower + weekly bullish
   - Short: RSI(7) > 75 + price > BB upper + weekly bearish
   - Mean-reversion at extremes works in range-bound 2025 market

5. ATR TRAILING STOP (2.5x):
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from 2022-style crashes

6. POSITION SIZING: 0.28 discrete (conservative for daily volatility)
   - BTC dropped 77% in 2022; 0.28 size = max -22% equity drawdown from single crash
   - Discrete levels minimize fee churn (each change costs 0.10%)

Why 1d should beat 12h (#353 Sharpe=-0.334):
- Daily closes are more significant (institutional level)
- Less noise than 12h, cleaner regime detection
- Weekly HMA provides stronger bias than daily HMA
- Should generate 15-30 trades/year (enough for stats, low fee drag)

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_adaptive_1w_hma_donchian_rsi_atr_v1"
timeframe = "1d"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 10:
        return adx
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            di_plus[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 1e-10:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / di_sum
    
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_series.values
    
    return adx

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_bb_width_percentile(upper, lower, sma, lookback=60):
    """Calculate BB Width as percentile of recent width values."""
    n = len(upper)
    bb_width = (upper - lower) / sma
    bb_width_pct = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if not np.isnan(bb_width[i]):
            recent_widths = bb_width[i-lookback:i+1]
            recent_widths = recent_widths[~np.isnan(recent_widths)]
            if len(recent_widths) > 0:
                bb_width_pct[i] = np.percentile(recent_widths, np.where(recent_widths <= bb_width[i])[0].size / len(recent_widths) * 100)
    
    return bb_width_pct

def calculate_rsi(close, period=7):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    rsi[np.isnan(rsi)] = 50.0
    return rsi

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_upper, bb_lower, bb_sma, 60)
    rsi = calculate_rsi(close, 7)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # Trending: ADX > 25
        trending_regime = adx[i] > 25
        
        # Ranging: ADX < 20 AND BB Width in lower 30th percentile
        ranging_regime = adx[i] < 20 and (not np.isnan(bb_width_pct[i]) and bb_width_pct[i] < 30)
        
        # === WEEKLY HMA BIAS ===
        bull_bias_1w = close[i] > hma_1w_aligned[i]
        bear_bias_1w = close[i] < hma_1w_aligned[i]
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # TRENDING REGIME: Breakout entries
        if trending_regime:
            # Long breakout: price breaks Donchian upper + weekly bullish
            if close[i] > donchian_upper[i-1] and bull_bias_1w:
                new_signal = SIZE
            
            # Short breakout: price breaks Donchian lower + weekly bearish
            elif close[i] < donchian_lower[i-1] and bear_bias_1w:
                new_signal = -SIZE
        
        # RANGING REGIME: Mean-reversion entries
        elif ranging_regime:
            # Long: RSI oversold + price at BB lower + weekly bullish
            if rsi[i] < 25 and close[i] < bb_lower[i] and bull_bias_1w:
                new_signal = SIZE
            
            # Short: RSI overbought + price at BB upper + weekly bearish
            elif rsi[i] > 75 and close[i] > bb_upper[i] and bear_bias_1w:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME FLIP EXIT ===
        # Exit if regime changes against position type
        if in_position and new_signal != 0.0:
            # Long position in newly ranging market without mean-revert signal
            if position_side > 0 and ranging_regime and not (rsi[i] < 25 and close[i] < bb_lower[i]):
                new_signal = 0.0
            # Short position in newly trending market without breakout signal
            if position_side < 0 and trending_regime and not (close[i] < donchian_lower[i-1]):
                new_signal = 0.0
        
        # === WEEKLY BIAS FLIP EXIT ===
        # Exit if weekly trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias_1w:
                new_signal = 0.0
            if position_side < 0 and bull_bias_1w:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals