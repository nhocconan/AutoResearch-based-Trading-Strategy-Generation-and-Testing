#!/usr/bin/env python3
"""
Experiment #359: 12h Dual-Regime Donchian Breakout with 1d HMA Bias + Volume Filter

Hypothesis: After analyzing 358 failed experiments, the pattern shows that single-regime 
strategies fail because they don't adapt. For 12h timeframe:

1. DUAL REGIME DETECTION: Use ADX + Bollinger Width to detect market state
   - ADX > 20 + BB Width expanding = TREND regime (use breakout logic)
   - ADX < 20 + BB Width contracting = RANGE regime (use mean reversion logic)
   - This adapts to both bull/bear/range markets dynamically

2. DONCHIAN BREAKOUT (20-period) for TREND regime:
   - Long: price breaks Donchian upper + volume > 1.5x avg + 1d HMA bullish
   - Short: price breaks Donchian lower + volume > 1.5x avg + 1d HMA bearish
   - Volume confirmation filters 40%+ of false breakouts

3. RSI MEAN REVERSION (14) for RANGE regime:
   - Long: RSI < 30 + price < BB lower + 1d HMA neutral/bullish
   - Short: RSI > 70 + price > BB upper + 1d HMA neutral/bearish
   - Captures reversals when trend logic would whipsaw

4. 1d HMA TREND BIAS (softer than #353):
   - Bullish: price > HMA * 0.995 (within 0.5% counts as bullish)
   - Bearish: price < HMA * 1.005 (within 0.5% counts as bearish)
   - Less strict = more trades while maintaining edge

5. POSITION SIZING: 0.30 discrete (higher than #353's 0.25 to generate more P&L)
   - Still conservative (max 30% capital)
   - Discrete levels minimize fee churn

6. STOPLOSS: 2.0 * ATR(14) trailing (tighter than #353's 2.5x)
   - Cut losses faster in 12h timeframe
   - Trail on profitable positions

Why this should beat #353 (Sharpe=-0.334):
- Dual regime = trades in both trending AND ranging markets
- Volume filter = fewer false breakouts, higher win rate
- Softer 1d HMA bias = more entry opportunities
- RSI mean reversion backup = catches reversals breakout misses
- Target: 30-50 trades/year per symbol (enough for statistical significance)

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_donchian_1d_hma_vol_rsi_atr_v1"
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
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    # Bandwidth = (upper - lower) / sma
    bandwidth = np.zeros(len(close))
    for i in range(period - 1, len(close)):
        if sma[i] > 1e-10:
            bandwidth[i] = (upper[i] - lower[i]) / sma[i]
    return upper, lower, bandwidth

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
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

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    volume_ma = calculate_volume_ma(volume, 20)
    
    # Calculate BB bandwidth percentile for regime detection
    bb_bw_percentile = np.full(n, np.nan)
    lookback = 50
    for i in range(lookback - 1, n):
        valid_bw = bb_bandwidth[i-lookback+1:i+1]
        valid_bw = valid_bw[~np.isnan(valid_bw)]
        if len(valid_bw) > 0:
            bb_bw_percentile[i] = np.sum(bb_bandwidth[i] > valid_bw) / len(valid_bw)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_bandwidth[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(volume_ma[i]) or volume_ma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === 1d HMA TREND BIAS (softer than #353) ===
        hma_threshold = 0.005  # 0.5% tolerance
        bull_trend_1d = close[i] > hma_1d_aligned[i] * (1 - hma_threshold)
        bear_trend_1d = close[i] < hma_1d_aligned[i] * (1 + hma_threshold)
        neutral_trend_1d = not bull_trend_1d and not bear_trend_1d
        
        # === REGIME DETECTION ===
        # TREND regime: ADX > 18 + BB bandwidth expanding (percentile > 0.5)
        trend_regime = adx[i] > 18 and (np.isnan(bb_bw_percentile[i]) or bb_bw_percentile[i] > 0.4)
        
        # RANGE regime: ADX < 18 + BB bandwidth contracting
        range_regime = adx[i] < 18 and (np.isnan(bb_bw_percentile[i]) or bb_bw_percentile[i] <= 0.5)
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]  # 30% above average
        
        # === DONCHIAN BREAKOUT SIGNALS (for TREND regime) ===
        long_breakout = close[i] > donchian_upper[i-1] if i > 0 else False
        short_breakout = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === RSI MEAN REVERSION SIGNALS (for RANGE regime) ===
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        price_at_bb_lower = close[i] <= bb_lower[i] * 1.002  # within 0.2%
        price_at_bb_upper = close[i] >= bb_upper[i] * 0.998  # within 0.2%
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY - TREND REGIME: Breakout + volume + 1d bullish bias
        if trend_regime and long_breakout and volume_confirmed and bull_trend_1d:
            new_signal = SIZE
        
        # SHORT ENTRY - TREND REGIME: Breakout + volume + 1d bearish bias
        elif trend_regime and short_breakout and volume_confirmed and bear_trend_1d:
            new_signal = -SIZE
        
        # LONG ENTRY - RANGE REGIME: RSI oversold + BB lower + 1d not bearish
        elif range_regime and rsi_oversold and price_at_bb_lower and not bear_trend_1d:
            new_signal = SIZE
        
        # SHORT ENTRY - RANGE REGIME: RSI overbought + BB upper + 1d not bullish
        elif range_regime and rsi_overbought and price_at_bb_upper and not bull_trend_1d:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 1d trend flips strongly against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and close[i] < hma_1d_aligned[i] * 0.98:  # 2% below HMA
                new_signal = 0.0
            if position_side < 0 and close[i] > hma_1d_aligned[i] * 1.02:  # 2% above HMA
                new_signal = 0.0
        
        # === REGIME CHANGE EXIT ===
        # Exit long if regime changes from trend to range (and vice versa for short)
        if in_position and position_side > 0 and range_regime and rsi[i] > 55:
            new_signal = 0.0
        if in_position and position_side < 0 and range_regime and rsi[i] < 45:
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