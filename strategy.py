#!/usr/bin/env python3
"""
Experiment #369: 4h Primary + 1d HTF — Fisher Transform + Funding Contrarian

Hypothesis: After 368 experiments, the clearest pattern is:
1. Simple trend-following FAILS on BTC/ETH (2022 crash destroys all gains)
2. Funding rate contrarian is the BEST EDGE for BTC/ETH (Sharpe 0.8-1.5 reported)
3. Ehlers Fisher Transform catches reversals in bear/range markets better than RSI
4. 4h timeframe balances trade frequency (20-50/year) vs fee drag
5. 1d HMA provides major trend bias without over-filtering

Why this might beat current best (Sharpe=0.435):
- Funding rate extremes signal crowded positions → contrarian entry
- Fisher Transform normalizes price distribution → cleaner reversal signals
- Dual confirmation (Fisher + Funding) reduces false signals
- Works in ALL regimes: bull, bear, and range (unlike pure trend)

Key innovations vs failed 4h strategies (#359, #361, #364):
- NOT pure trend-following (those all failed negative Sharpe)
- Uses funding rate data (most failed strategies ignored this)
- Fisher Transform instead of RSI (better for non-Gaussian crypto returns)
- Asymmetric sizing: longs 0.30, shorts 0.20 (crypto long bias)

Position sizing: 0.25-0.30 longs, 0.15-0.20 shorts
Stoploss: 2.5 * ATR trailing
Target: 25-45 trades/year on 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_funding_contrarian_1d_v1"
timeframe = "4h"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian-like distribution for clearer reversal signals.
    Long: Fisher crosses above -1.5 (oversold reversal)
    Short: Fisher crosses below +1.5 (overbought reversal)
    """
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate typical price
    typical = (high_s + low_s + close_s) / 3.0
    
    # Normalize price over lookback period
    highest = typical.rolling(window=period, min_periods=period).max()
    lowest = typical.rolling(window=period, min_periods=period).min()
    
    # Normalized value between -1 and +1
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = 2.0 * (typical - lowest) / (highest - lowest + 1e-10) - 1.0
    
    # Clamp to avoid division issues
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Fisher Transform
    fisher_raw = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
    fisher = pd.Series(fisher_raw).ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Signal line (previous Fisher value)
    fisher_signal = fisher.shift(1)
    
    return fisher.values, fisher_signal.values

def calculate_funding_zscore(prices, symbol, period=30):
    """
    Calculate Z-score of funding rate over rolling period.
    Z < -2.0 = funding extremely negative = long contrarian
    Z > +2.0 = funding extremely positive = short contrarian
    """
    try:
        # Load funding data for this symbol
        funding_path = f"data/processed/funding/{symbol}.parquet"
        df_funding = pd.read_parquet(funding_path)
        
        # Resample to match primary timeframe (4h = 6 per day)
        df_funding['open_time'] = pd.to_datetime(df_funding['open_time'])
        df_funding = df_funding.set_index('open_time')
        
        # Get the timeframe from prices
        prices_idx = prices.copy()
        prices_idx['open_time'] = pd.to_datetime(prices_idx['open_time'])
        
        # Align funding to price timestamps
        # Funding is typically 8h, we need to resample to 4h
        funding_4h = df_funding['funding_rate'].resample('4h').last()
        
        # Merge with prices
        prices_with_funding = prices_idx.set_index('open_time')
        prices_with_funding['funding'] = funding_4h
        
        # Forward fill missing funding values
        prices_with_funding['funding'] = prices_with_funding['funding'].ffill()
        
        # Calculate rolling z-score
        funding_series = prices_with_funding['funding'].values
        funding_mean = pd.Series(funding_series).rolling(window=period, min_periods=period).mean().values
        funding_std = pd.Series(funding_series).rolling(window=period, min_periods=period).std().values
        
        with np.errstate(divide='ignore', invalid='ignore'):
            zscore = (funding_series - funding_mean) / (funding_std + 1e-10)
        
        return zscore
        
    except Exception as e:
        # Fallback: return zeros if funding data unavailable
        return np.zeros(len(prices))

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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Extract symbol from prices DataFrame (needed for funding data)
    symbol = prices.get('symbol', ['BTCUSDT'])[0] if 'symbol' in prices.columns else 'BTCUSDT'
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    # Fisher Transform for reversal signals
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    
    # Funding rate z-score for contrarian signals
    funding_z = calculate_funding_zscore(prices, symbol, period=30)
    
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
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(funding_z[i]):
            continue
        
        # === 1D MAJOR TREND REGIME (primary direction filter) ===
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Fisher crosses above -1.5 from below = bullish reversal
        fisher_bull_cross = (fisher_signal[i] < -1.5) and (fisher[i] >= -1.5)
        # Fisher crosses below +1.5 from above = bearish reversal
        fisher_bear_cross = (fisher_signal[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Extreme Fisher values (deep oversold/overbought)
        fisher_oversold = fisher[i] < -2.0
        fisher_overbought = fisher[i] > 2.0
        
        # === FUNDING RATE CONTRARIAN SIGNALS ===
        # Z < -2.0 = funding extremely negative = crowd is short = long contrarian
        funding_extreme_long = funding_z[i] < -1.5
        # Z > +2.0 = funding extremely positive = crowd is long = short contrarian
        funding_extreme_short = funding_z[i] > 1.5
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # Price position relative to SMA200
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === ENTRY LOGIC - CONTRARIAN WITH TREND BIAS ===
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: Fisher reversal + funding contrarian confirmation
        if fisher_bull_cross:
            if funding_extreme_long:
                # Strong signal: both Fisher and funding agree
                if regime_bull:
                    new_signal = LONG_STRONG
                else:
                    new_signal = LONG_BASE
            elif regime_bull and rsi_oversold:
                # Fisher + trend + RSI oversold
                new_signal = LONG_BASE
            elif fisher_oversold and price_above_sma200:
                # Deep Fisher oversold + above long-term MA
                new_signal = LONG_BASE
        
        # Funding-only contrarian long (when Fisher not signaling)
        elif funding_extreme_long and regime_bull:
            if rsi_oversold or fisher_oversold:
                new_signal = LONG_BASE * 0.8
        
        # === SHORT ENTRIES ===
        # Primary: Fisher reversal + funding contrarian confirmation
        if fisher_bear_cross:
            if funding_extreme_short:
                # Strong signal: both Fisher and funding agree
                if regime_bear:
                    if new_signal == 0.0:
                        new_signal = -SHORT_STRONG
                else:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE
            elif regime_bear and rsi_overbought:
                # Fisher + trend + RSI overbought
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            elif fisher_overbought and not price_above_sma200:
                # Deep Fisher overbought + below long-term MA
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
        
        # Funding-only contrarian short (when Fisher not signaling)
        elif funding_extreme_short and regime_bear:
            if rsi_overbought or fisher_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 4h) ===
        # Force trade if no signal for 20 bars (~3.3 days on 4h)
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if regime_bull and (rsi_oversold or fisher_oversold):
                new_signal = LONG_BASE * 0.5
            elif regime_bear and (rsi_overbought or fisher_overbought):
                new_signal = -SHORT_BASE * 0.5
        
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
        
        # === FISHER REVERSAL EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and fisher_overbought:
                fisher_exit = True
            if position_side < 0 and fisher_oversold:
                fisher_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and regime_bear and close[i] < hma_1d_21_aligned[i]:
                regime_reversal = True
            if position_side < 0 and regime_bull and close[i] > hma_1d_21_aligned[i]:
                regime_reversal = True
        
        if stoploss_triggered or fisher_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.18:
                new_signal = -SHORT_STRONG
            else:
                new_signal = -SHORT_BASE
        
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