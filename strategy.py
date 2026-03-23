#!/usr/bin/env python3
"""
Experiment #698: 30m Primary + 4h/1d HTF — Keltner/BB Squeeze + StochRSI + HTF Trend

Hypothesis: After 608 failed strategies, the pattern for lower TF (30m) is clear:
1. CHOP+CRSI has failed 50+ times — need DIFFERENT indicator combination
2. #695 (1h Fisher+BB) got Return=+16.4% but negative Sharpe — Fisher has merit
3. Lower TF needs VERY strict entry to avoid fee drag (>100 trades/yr = death)
4. Key insight: Use HTF (4h/1d) for DIRECTION, 30m only for ENTRY TIMING

This strategy uses:
- Keltner Channel + Bollinger Band squeeze for volatility regime (proven in literature)
- Stochastic RSI (14,14,3,3) for precise entry timing (more sensitive than RSI)
- 4h HMA(21) for trend direction bias (slower than EMA, less whipsaw)
- 1d ADX(14) for regime filter (trend vs range)
- Volume confirmation (>0.6x 20-bar avg) — loosened from 0.8x to generate trades

Why this might beat Sharpe=0.520:
- BB/KC squeeze is proven volatility breakout signal (John Carter, Squeeze Momentum)
- StochRSI catches entries earlier than regular RSI in mean-reversion
- 4h HMA gives trend bias without overfiltering (unlike 1d HMA which is too slow)
- Asymmetric entries: mean-revert in range, trend-pullback in trend
- Loosened entry conditions to ensure ≥10 trades/symbol (learned from #685 failure)

Position sizing: 0.20 (conservative for 30m TF per Rule 4, Rule 10)
Target: 40-80 trades/year on 30m (Rule 10 limit)
Stoploss: 2.0*ATR trailing (tighter for lower TF)

CRITICAL LESSONS FROM FAILURES:
- Entry conditions MUST generate trades (#685 = 0 trades = auto-reject)
- StochRSI <0.25 for long, >0.75 for short (not extreme 0.1/0.9)
- Squeeze is BONUS not requirement (either squeeze OR extreme StochRSI)
- No session filter (was killing trade count)
- Volume >0.6x (not 0.8x) to allow more entries
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_squeeze_stochrsi_hma4h_adx1d_v1"
timeframe = "30m"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    bandwidth = (upper - lower) / (sma + 1e-10)
    pct_b = (close - lower) / (upper - lower + 1e-10)
    
    return upper.values, lower.values, bandwidth.values, pct_b.values

def calculate_keltner_channels(high, low, close, period=20, multiplier=1.5):
    """Calculate Keltner Channels (EMA-based with ATR)."""
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    
    return upper.values, lower.values

def calculate_stochastic_rsi(close, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
    """
    Calculate Stochastic RSI.
    
    StochRSI = (RSI - Lowest RSI) / (Highest RSI - Lowest RSI)
    
    Signals:
    - Long: StochRSI < 0.20 (oversold)
    - Short: StochRSI > 0.80 (overbought)
    """
    close_s = pd.Series(close)
    
    # Calculate RSI
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # Calculate Stochastic of RSI
    lowest_rsi = rsi.rolling(window=stoch_period, min_periods=stoch_period).min()
    highest_rsi = rsi.rolling(window=stoch_period, min_periods=stoch_period).max()
    
    stoch_rsi = (rsi - lowest_rsi) / (highest_rsi - lowest_rsi + 1e-10)
    
    return stoch_rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d ADX for regime filter
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_bandwidth, bb_pct_b = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    kc_upper, kc_lower = calculate_keltner_channels(high, low, close, period=20, multiplier=1.5)
    stoch_rsi = calculate_stochastic_rsi(close, rsi_period=14, stoch_period=14, k_period=3, d_period=3)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40; Rule 10 - smaller for 30m)
    POSITION_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    # ADX hysteresis tracking
    prev_adx_regime = 0  # 0=neutral, 1=trend, 2=range
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(stoch_rsi[i]) or np.isnan(bb_pct_b[i]) or np.isnan(vol_sma[i]):
            continue
        if np.isnan(adx_1d_aligned[i]):
            continue
        if atr_14[i] == 0 or vol_sma[i] == 0:
            continue
        
        # === 4H TREND BIAS ===
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-5] if i >= 5 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-5] if i >= 5 else False
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1D ADX REGIME (with hysteresis) ===
        adx_val = adx_1d_aligned[i]
        
        # Hysteresis: trend regime needs ADX>25 to enter, <18 to exit
        if adx_val > 25.0:
            adx_regime = 1  # Trending
        elif adx_val < 18.0:
            adx_regime = 2  # Range
        else:
            adx_regime = prev_adx_regime  # Keep previous regime
        
        prev_adx_regime = adx_regime
        is_trend_regime = (adx_regime == 1)
        is_range_regime = (adx_regime == 2)
        
        # === SQUEEZE DETECTION (BB inside KC = low vol) ===
        squeeze_on = (bb_upper[i] <= kc_upper[i]) and (bb_lower[i] >= kc_lower[i])
        
        # === STOCHASTIC RSI SIGNALS ===
        stoch_rsi_oversold = stoch_rsi[i] < 0.25
        stoch_rsi_overbought = stoch_rsi[i] > 0.75
        stoch_rsi_extreme_low = stoch_rsi[i] < 0.15
        stoch_rsi_extreme_high = stoch_rsi[i] > 0.85
        
        # === BOLLINGER BAND SIGNALS ===
        bb_touch_lower = bb_pct_b[i] < 0.15
        bb_touch_upper = bb_pct_b[i] > 0.85
        
        # === VOLUME CONFIRMATION (loosened to 0.6x) ===
        volume_confirmed = volume[i] > 0.6 * vol_sma[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime 1: Range market (ADX < 18) + mean reversion
        if is_range_regime:
            # Either squeeze breakout OR extreme StochRSI
            mean_revert_signal = (stoch_rsi_extreme_low or bb_touch_lower)
            if mean_revert_signal and volume_confirmed:
                # Only long if 4h not strongly bearish
                if price_above_hma_4h or not hma_4h_slope_bear:
                    new_signal = POSITION_SIZE
        
        # Regime 2: Trending market (ADX > 25) + trend pullback
        elif is_trend_regime:
            if hma_4h_slope_bull and price_above_hma_4h:
                # Pullback entry in uptrend
                pullback_signal = stoch_rsi_oversold or bb_touch_lower
                if pullback_signal and volume_confirmed:
                    new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Regime 1: Range market (ADX < 18) + mean reversion
        if is_range_regime:
            mean_revert_signal = (stoch_rsi_extreme_high or bb_touch_upper)
            if mean_revert_signal and volume_confirmed:
                # Only short if 4h not strongly bullish
                if price_below_hma_4h or not hma_4h_slope_bull:
                    new_signal = -POSITION_SIZE
        
        # Regime 2: Trending market (ADX > 25) + trend pullback
        elif is_trend_regime:
            if hma_4h_slope_bear and price_below_hma_4h:
                # Pullback entry in downtrend
                pullback_signal = stoch_rsi_overbought or bb_touch_upper
                if pullback_signal and volume_confirmed:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_4h_slope_bear and price_below_hma_4h and is_trend_regime:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_4h_slope_bull and price_above_hma_4h and is_trend_regime:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals